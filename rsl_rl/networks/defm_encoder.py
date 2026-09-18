# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import os
import shutil
import time
import torch
import torch.distributed as dist
import torch.nn.functional as F
from math import prod

import torchvision.transforms.v2 as tt


def _encoder_compute_device() -> torch.device:
    """Device for DeFM weights and probe tensors; respects ``torch.cuda.set_device`` (DDP local rank)."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda", torch.cuda.current_device())


def _clear_defm_hub_cache(rank: int = 0) -> None:
    """Remove leggedrobotics defm cache dirs under torch hub so the next load does a fresh clone."""
    try:
        hub_dir = torch.hub.get_dir()
        if rank == 0:
            print(f"[Info] Using torch hub directory: {hub_dir}")
        if os.path.exists(hub_dir):
            for item in os.listdir(hub_dir):
                if item.startswith("leggedrobotics"):
                    cache_path = os.path.join(hub_dir, item)
                    if os.path.isdir(cache_path):
                        if rank == 0:
                            print(f"[Info] Removing cache directory: {cache_path}")
                        shutil.rmtree(cache_path, ignore_errors=True)
    except Exception as cleanup_error:
        if rank == 0:
            print(f"[Warning] Cache cleanup failed: {cleanup_error}")


def _load_defm_model_with_retry(model_name: str, ckpt_path: str | None = None, max_retries: int = 3, rank: int = 0):
    """
    Load DeFM model from torch.hub with retry logic to handle cache issues.

    Args:
        ckpt_path: Path to the checkpoint file.
        max_retries: Maximum number of retry attempts.
        rank: Process rank (for logging).

    Returns:
        Loaded DeFM model.

    Raises:
        RuntimeError: If loading fails after all retries.
    """
    for attempt in range(max_retries):
        try:
            model = torch.hub.load(
                "leggedrobotics/defm:main",
                model_name,
                pretrained=True,
                force_reload=False,
                trust_repo=True,
                pretrained_path=ckpt_path,
            )
            return model

        except (OSError, FileNotFoundError) as e:
            err_str = str(e)
            # Incomplete/corrupt cache: dir exists but e.g. hubconf.py is missing
            is_bad_cache = (
                "hubconf.py" in err_str
                or "Directory not empty" in err_str
                or (isinstance(e, FileNotFoundError) and "leggedrobotics_defm" in err_str)
            )
            if is_bad_cache and attempt < max_retries - 1:
                if rank == 0:
                    print(f"[Warning] torch.hub cache issue detected (attempt {attempt + 1}/{max_retries}): {e}")
                    print("[Info] Cleaning up torch.hub cache and retrying...")
                    _clear_defm_hub_cache(rank=0)
                # Only rank 0 clears cache to avoid distributed races; others just retry after a short wait
                time.sleep(1)
            else:
                raise RuntimeError(f"Failed to load DeFM model after {max_retries} attempts on rank {rank}") from e

        except Exception as e:
            if attempt < max_retries - 1:
                if rank == 0:
                    print(f"[Warning] Error loading model (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(1)
            else:
                raise RuntimeError(f"Failed to load DeFM model after {max_retries} attempts on rank {rank}") from e

    raise RuntimeError(f"Failed to load DeFM model after {max_retries} attempts on rank {rank}")


class DepthImageEncoder(torch.nn.Module):
    """
    A depth image encoder that encodes the depth image into a feature vector.
    """

    def __init__(
        self,
        model_name: str,
        ckpt_path: str | None = None,
        freeze_backbone: bool = True,
        flatten_features: bool = False,
        use_half_precision: bool = False,
        compile_model: bool = False,
        target_size: tuple[int, int] | None = (64, 96),
        input_size: tuple[int, int] | None = (60, 96),
        cnn_padding: bool = True,
    ):
        """
        Initialize the DepthImageEncoder.

        Args:
            model_name: Name of the DeFM model to load.
            ckpt_path: Path to the checkpoint file.
            freeze_backbone: Whether to freeze the backbone of the RegNet student.
            flatten_features: Whether to use the cls token. Default is False.
            use_half_precision: Whether to use FP16 for faster inference (only works with freeze_backbone=True).
            compile_model: Whether to use torch.compile for optimization (PyTorch 2.0+).
        """
        super().__init__()

        self.flatten_features = flatten_features
        self.freeze_backbone = freeze_backbone
        self.use_half_precision = use_half_precision
        self.compile_model = compile_model
        self.target_size = target_size
        self.cnn_padding = cnn_padding

        # constants for preprocessing
        self.max_depth_c1 = 100.0
        self.max_depth_c2 = 9.0
        self.mean = [0.248880, 0.495620, 0.492858]
        self.std = [0.139357, 0.271314, 0.297177]

        self.normalize = tt.Normalize(mean=self.mean, std=self.std)

        if self.cnn_padding:
            # Pad to next multiple of 32 (Since this is needed for the BiFPNs)
            self.pad_h = (32 - (input_size[0] % 32)) % 32
            self.pad_w = (32 - (input_size[1] % 32)) % 32

        if use_half_precision and not freeze_backbone:
            raise ValueError("Half precision inference requires freeze_backbone=True for stability")

        if ckpt_path is not None and not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

        # Torch hub: avoid concurrent clone/extract into the same cache directory on one host.
        # - Single-host DDP: LOCAL_RANK==0 loads first, then barrier, then other locals (original behavior).
        # - Multi-node: each host must populate its own hub cache once before other GPUs on that host load;
        #   using global rank==0 only would leave other nodes' caches empty and let their ranks race.
        is_distributed = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_distributed else 0
        if is_distributed:
            _lr = os.environ.get("LOCAL_RANK")
            hub_leader = int(_lr) == 0 if _lr is not None else rank == 0
        else:
            hub_leader = True

        if hub_leader:
            print(f"[Rank {rank}] Loading DeFM model from: {model_name} with checkpoint: {ckpt_path}")

        if hub_leader:
            self.model = _load_defm_model_with_retry(model_name, ckpt_path, max_retries=5, rank=rank)

        if is_distributed:
            dist.barrier()

        if is_distributed and not hub_leader:
            self.model = _load_defm_model_with_retry(model_name, ckpt_path, max_retries=5, rank=rank)

        _device = _encoder_compute_device()
        self.model.eval().to(_device)

        # Do a forward pass to get model output size and shape
        with torch.no_grad():
            output = self.forward(torch.zeros(1, 1, input_size[0], input_size[1], device=_device, dtype=torch.float32))
            self._output_dims = tuple(output.shape[1:])

        if rank == 0:
            print(
                "Loaded model: 'defm_regnet_y_400mf' with "
                f"{sum(p.numel() for p in self.model.parameters())/1e6:.2f}M parameters."
            )

        # Freeze the DeFM model
        if self.freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

            # Apply optimizations for frozen inference
            if self.use_half_precision:
                if rank == 0:
                    print("[Optimization] Converting model to FP16 for faster inference...")
                self.model = self.model.half()

            if self.compile_model:
                if rank == 0:
                    print("[Optimization] Compiling model with torch.compile (this may take a moment)...")
                try:
                    # Use 'reduce-overhead' mode for best inference performance
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    if rank == 0:
                        print("[Optimization] Model compilation successful!")
                except Exception as e:
                    if rank == 0:
                        print(f"[Warning] torch.compile failed: {e}. Continuing without compilation.")
                        print("[Info] Make sure you're using PyTorch 2.0+ for torch.compile support.")

    """
    Properties.
    """

    @property
    def output_dims(self) -> tuple[int, ...]:
        """Output dimensions as a tuple."""
        return self._output_dims

    @property
    def output_size(self) -> int:
        """Output size as a product of the output dimensions."""
        return prod(self._output_dims)

    """
    Operations.
    """

    def init_weights(self):
        pass

    def reset(self, dones=None, hidden_states=None):
        pass

    def detach_hidden_states(self, dones=None):
        pass

    def forward(self, depth_image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the depth image encoder.

        Args:
            depth_image: Input depth image tensor of shape (B, 1, 60, 106).
            add_noise: Whether to add noise to the depth image.

        Returns:
            Encoded features of shape (B, 64, 6, 10) if flatten_features is False, otherwise (B, 64 * 6 * 10 + 64).
        """
        if len(depth_image.shape) == 3:
            depth_image = depth_image.unsqueeze(1)

        # Preprocess the depth image
        preprocessed_depth = self._preprocess_depth(depth_image)

        # Convert to half precision if enabled
        if self.use_half_precision and self.freeze_backbone:
            preprocessed_depth = preprocessed_depth.half()

        # Forward pass through the model
        # Use inference_mode for frozen models (faster than no_grad)
        if self.freeze_backbone:
            with torch.inference_mode():
                output = self.model(preprocessed_depth)
        else:
            output = self.model(preprocessed_depth)

        # Extract and convert only the needed outputs to float32
        if self.flatten_features:
            # print(f"Global backbone shape: {output['global_backbone'].shape}")
            # print(f"Dense BIFPN P4 shape: {output['dense_bifpn']['P4'].shape}")
            p4_features = output["dense_bifpn"]["P4"]
            global_features = output["global_backbone"]

            # Convert to float32 only if using half precision
            if self.use_half_precision and self.freeze_backbone:
                p4_features = p4_features.float()
                global_features = global_features.float()
            # flattened features: (B, C_p4 * H_p4 * W_p4 + C_global)
            return torch.cat([torch.flatten(p4_features, start_dim=1), global_features], dim=1)
        else:
            p4_features = output["dense_bifpn"]["P4"]

            # Convert to float32 only if using half precision
            if self.use_half_precision and self.freeze_backbone:
                p4_features = p4_features.float()

            # unflattened features: (B, C_p4, H_p4, W_p4)
            return p4_features

    def _preprocess_depth(self, depth: torch.Tensor) -> torch.Tensor:
        # depth shape: (B, 1, H, W)
        t = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

        # Clip to max depth for C1 computation
        t_clamped = torch.clamp(t, min=0.0, max=self.max_depth_c1)
        log_depth = torch.log1p(t_clamped)

        # C1
        c1 = log_depth / math.log1p(self.max_depth_c1)
        # C2
        c2 = torch.clamp(log_depth / math.log1p(self.max_depth_c2), min=0.0, max=1.0)
        # C3
        B, _, H, W = log_depth.shape
        log_flat = log_depth.view(B, -1)
        min_log = log_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        max_log = log_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        denom = max_log - min_log

        denom_safe = torch.where(denom > 0.0, denom, torch.ones_like(denom))
        c3 = (log_depth - min_log) / denom_safe
        c3 = torch.where(denom > 0.0, c3, torch.zeros_like(c3))

        # Stack
        x = torch.cat([c1, c2, c3], dim=1)

        # Resize/Pad (Here using interpolation as nearest multiple size for testing)
        if not self.cnn_padding:
            x = F.interpolate(x, size=self.target_size, mode="nearest")

        # Normalize via torchvision
        x = self.normalize(x)

        # Pad if CNN padding is enabled
        if self.cnn_padding:
            x = F.pad(x, (0, self.pad_w, 0, self.pad_h), mode="constant", value=0.0)

        return x


if __name__ == "__main__":
    from rsl_rl import RSL_RL_RESOURCES_PATH

    ckpt_path = os.path.join(RSL_RL_RESOURCES_PATH, "defm_regnet_y_400mf_112.pth")

    # --------- Prepare dummy input ---------
    batch_size = 256
    dummy_depth = torch.randn(batch_size, 1, 60, 106) * 10.0  # Simulate depth values in meters
    dummy_depth = torch.clamp(dummy_depth, min=0.1, max=10.0).to("cuda")  # Clamp to valid depth range

    # --------- Benchmark different configurations ---------
    configs = [
        {"name": "Baseline", "use_half_precision": False, "compile_model": False},
        {"name": "FP16", "use_half_precision": True, "compile_model": False},
        {"name": "torch.compile", "use_half_precision": False, "compile_model": True},
        {"name": "FP16 + torch.compile", "use_half_precision": True, "compile_model": True},
    ]

    num_iterations = 100
    num_warmup = 10

    print("\n" + "=" * 80)
    print("DEFM ENCODER INFERENCE BENCHMARK")
    print("=" * 80)

    baseline_output = None

    for config in configs:
        print(f"\n🚀 Testing: {config['name']}")
        print("-" * 80)

        # Initialize model
        model = DepthImageEncoder(
            # ckpt_path=ckpt_path,
            # model_name="defm_regnet_y_400mf",
            model_name="defm_resnet18",
            flatten_features=True,
            use_half_precision=config["use_half_precision"],
            compile_model=config["compile_model"],
        )

        # Warmup
        print(f"Warming up ({num_warmup} iterations)...")
        for _ in range(num_warmup):
            _ = model(dummy_depth)

        # Get a reference output for validation
        with torch.inference_mode():
            validation_output = model(dummy_depth)

        # Benchmark
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(num_iterations):
            output = model(dummy_depth)
        torch.cuda.synchronize()
        end_time = time.time()

        avg_time_ms = (end_time - start_time) * 1000 / num_iterations
        throughput = (batch_size * num_iterations) / (end_time - start_time)

        print(f"✓ Output shape: {output.shape}")
        print(f"✓ Average time: {avg_time_ms:.2f} ms/batch")
        print(f"✓ Throughput:   {throughput:.1f} images/sec")
        print(f"✓ FPS per image: {1000.0 / (avg_time_ms / batch_size):.1f}")

        # Validate outputs against baseline
        if baseline_output is None:
            baseline_output = validation_output.clone()
            print("✓ Baseline established")
        else:
            # Compare with baseline
            max_diff = (validation_output - baseline_output).abs().max().item()
            mean_diff = (validation_output - baseline_output).abs().mean().item()
            relative_error = (
                ((validation_output - baseline_output).abs() / (baseline_output.abs() + 1e-8)).mean().item()
            )

            print(f"✓ Max difference from baseline:  {max_diff:.6f}")
            print(f"✓ Mean difference from baseline: {mean_diff:.6f}")
            print(f"✓ Relative error:                {relative_error:.6f} ({relative_error*100:.4f}%)")

            # Warn if differences are large
            if relative_error > 0.01:  # >1% relative error
                print("⚠️  WARNING: Large deviation from baseline!")
            elif relative_error > 0.001:  # >0.1% relative error
                print("⚠️  Note: Small deviation detected (acceptable for FP16)")

        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 80 + "\n")