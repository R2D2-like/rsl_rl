# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import statistics
import time
import torch
import warnings
from collections import deque
import gc
import pickle
import json
import h5py
from pathlib import Path

import rsl_rl
from rl_reach.tasks.reach.config.agents.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import resolve_rnd_config, resolve_symmetry_config
from rl_reach.tasks.reach.config.agents.actor_critic import ActorCritic
from rsl_rl.utils import resolve_obs_groups, store_code_state


class OnPolicyRunner:
    """On-policy runner for training and evaluation of actor-critic methods."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # check if multi-gpu is enabled
        self._configure_multi_gpu()

        # store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.profiler_enabled = self.cfg.get("profiler_enabled", False)

        # query observations from environment for algorithm construction
        obs = self.env.get_observations()
        default_sets = ["critic"]
        if "rnd_cfg" in self.alg_cfg and self.alg_cfg["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        self.cfg["obs_groups"] = resolve_obs_groups(obs, self.cfg["obs_groups"], default_sets)

        # create the algorithm
        self.alg = self._construct_algorithm(obs)

        # Decide whether to disable logging
        # We only log from the process with rank 0 (main process)
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0

        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        # initialize writer
        self._prepare_logging_writer()

        # randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # start learning
        obs = self.env.get_observations().to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # create buffers for logging extrinsic and intrinsic rewards
        if self.alg.rnd:
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Start training
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations

        # Initial memory snapshot before training
        if not self.disable_logs and self.log_dir is not None and self.profiler_enabled:
            print("\nInitial memory state before training:")
            self.print_tensor_memory_snapshot(start_iter)

        # Initialize profiler for memory and performance analysis
        # Note: Using conservative settings to avoid conflicts with Isaac Sim
        if self.profiler_enabled and not self.disable_logs and self.log_dir is not None:
            prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CUDA,
                    torch.profiler.ProfilerActivity.CPU
                ],
                schedule=torch.profiler.schedule(
                    wait=2,     # Skip first 2 iterations (reduced for stability)
                    warmup=1,   # Warmup for 1 iteration
                    active=2,   # Profile for 2 iterations
                    repeat=1    # Run once to avoid issues
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    os.path.join(self.log_dir, 'profiler_logs')
                ),
                record_shapes=True,        # Records tensor shapes
                profile_memory=True,       # Tracks memory usage
                with_stack=False,          # Disable stack traces (can cause issues)
                with_flops=False           # Disable flops (not essential)
            )
            prof.start()

        try:
            for it in range(start_iter, tot_iter):
                self.env.unwrapped.iteration = int(it)
                start = time.time()
                # Rollout
                with torch.inference_mode():
                    for step in range(self.num_steps_per_env):
                        # Sample actions
                        actions = self.alg.act(obs)
                        # Step the environment
                        obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                        # Move to device
                        obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                        # process the step
                        self.alg.process_env_step(obs, rewards, dones, extras)
                        # Extract intrinsic rewards (only for logging)
                        rnd_active = (
                            hasattr(self.alg, "rnd")
                            and self.alg.rnd is not None
                            and getattr(self.alg, "num_updates", 0)
                            > getattr(self.alg, "freeze_actor_iterations", -1)
                        )
                        intrinsic_rewards = self.alg.intrinsic_rewards if rnd_active else None
                        # book keeping
                        if self.log_dir is not None:
                            if "episode" in extras:
                                ep_infos.append(extras["episode"])
                            elif "log" in extras:
                                ep_infos.append(extras["log"])
                            # Update rewards
                            if rnd_active:
                                cur_ereward_sum += rewards
                                cur_ireward_sum += intrinsic_rewards  # type: ignore
                                cur_reward_sum += rewards + intrinsic_rewards
                            else:
                                cur_reward_sum += rewards
                            # Update episode length
                            cur_episode_length += 1
                            # Clear data for completed episodes
                            # -- common
                            new_ids = (dones > 0).nonzero(as_tuple=False)
                            rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                            cur_reward_sum[new_ids] = 0
                            cur_episode_length[new_ids] = 0
                            # -- intrinsic and extrinsic rewards
                            if rnd_active:
                                erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                                irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                                cur_ereward_sum[new_ids] = 0
                                cur_ireward_sum[new_ids] = 0
                            # self._collect_observation_data(obs, self.current_learning_iteration, step)

                stop = time.time()
                collection_time = stop - start
                start = stop

                # compute returns
                self.alg.compute_returns(obs)

                # update policy
                loss_dict = self.alg.update()
                # torch.cuda.empty_cache()
                # self._collect_observation_data(obs, self.current_learning_iteration)

                stop = time.time()
                learn_time = stop - start
                self.current_learning_iteration = it
                # log info
                if self.log_dir is not None and not self.disable_logs:
                    # Log information
                    self.log(locals())
                    # Save model
                    if it % self.save_interval == 0:
                        self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

                # Clear episode infos
                ep_infos.clear()
                # torch.cuda.empty_cache()

                # Memory snapshots at regular intervals
                if not self.disable_logs and self.log_dir is not None and self.profiler_enabled:
                    # Print memory snapshot every 100 iterations
                    if it % 100 == 0:
                        self.print_tensor_memory_snapshot(it)

                    # Save detailed memory snapshot every 500 iterations
                    if it % 500 == 0:
                        self.save_memory_snapshot(it)
                    prof.step()

                # Save code state
                if it == start_iter and not self.disable_logs:
                    # obtain all the diff files
                    git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                    # if possible store them to wandb
                    if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                        for path in git_file_paths:
                            self.writer.save_file(path)
        finally:
            # Stop profiler if it was started
            if self.profiler_enabled:
                prof.stop()

        # Save the final model after training
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        # Compute the collection size
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        # Update total time-steps and time
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # -- Episode info
        ep_string = ""
        if locs["ep_infos"]:
            # union of keys over all infos (some keys, e.g. time-out-only metrics, are not present at every step)
            ep_keys = list(dict.fromkeys(key for ep_info in locs["ep_infos"] for key in ep_info))
            for key in ep_keys:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # log to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        mean_std = self.alg.policy.action_std.mean()
        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

        # ------------------------------------------------------------
        # Camera exploration mode std logging
        # ------------------------------------------------------------
        arm_mean_std = None
        wrist_mean_std = None

        if (
            hasattr(self.alg.policy, "use_camera_exploration_mode")
            and self.alg.policy.use_camera_exploration_mode
            and hasattr(self.alg.policy, "arm_joint_ids")
            and hasattr(self.alg.policy, "wrist_joint_ids")
        ):
            action_std = self.alg.policy.action_std

            arm_mean_std = (
                action_std[:, self.alg.policy.arm_joint_ids]
                .mean()
            )

            wrist_mean_std = (
                action_std[:, self.alg.policy.wrist_joint_ids]
                .mean()
            )

        # -- Losses
        for key, value in locs["loss_dict"].items():
            self.writer.add_scalar(f"Loss/{key}", value, locs["it"])
        # Log learning rate(s).
        # - legacy algorithms may expose a single scalar `learning_rate`
        # - PointNetPPO may expose separate actor/critic learning rates and set
        #   `self.alg.learning_rate = None`
        if getattr(self.alg, "learning_rate", None) is not None:
            self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        if hasattr(self.alg, "actor_learning_rate"):
            self.writer.add_scalar("Loss/actor_learning_rate", self.alg.actor_learning_rate, locs["it"])
        if hasattr(self.alg, "critic_learning_rate"):
            self.writer.add_scalar("Loss/critic_learning_rate", self.alg.critic_learning_rate, locs["it"])

        # Log PointNet learning rate separately if using PointNetPPO
        if hasattr(self.alg, 'has_pointnet') and self.alg.has_pointnet:
            # Extract PointNet learning rate from the first parameter group
            pointnet_lr = self.alg.optimizer.param_groups[0]['lr']
            self.writer.add_scalar("Loss/pointnet_learning_rate", pointnet_lr, locs["it"])

        # -- Policy
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])

        if arm_mean_std is not None:
            self.writer.add_scalar(
                "Policy/arm_mean_std",
                arm_mean_std.item(),
                locs["it"],
            )

        if wrist_mean_std is not None:
            self.writer.add_scalar(
                "Policy/wrist_mean_std",
                wrist_mean_std.item(),
                locs["it"],
            )

        # -- Performance
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # -- Training
        if len(locs["rewbuffer"]) > 0:
            rnd_active_for_log = (
                hasattr(self.alg, "rnd")
                and self.alg.rnd is not None
                and getattr(self.alg, "num_updates", 0)
                > getattr(self.alg, "freeze_actor_iterations", -1)
                and "erewbuffer" in locs
                and "irewbuffer" in locs
                and len(locs["erewbuffer"]) > 0
                and len(locs["irewbuffer"]) > 0
            )
            # separate logging for intrinsic and extrinsic rewards
            if rnd_active_for_log:
                self.writer.add_scalar("Rnd/mean_extrinsic_reward", statistics.mean(locs["erewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/mean_intrinsic_reward", statistics.mean(locs["irewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/weight", self.alg.rnd.weight, locs["it"])
            # everything else
            self.writer.add_scalar("Episode_Reward/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Episode_Termination/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )

            if arm_mean_std is not None:
                log_string += (
                    f"""{'Arm action noise std:':>{pad}} {arm_mean_std.item():.4f}\n"""
                )

            if wrist_mean_std is not None:
                log_string += (
                    f"""{'Wrist action noise std:':>{pad}} {wrist_mean_std.item():.4f}\n"""
                )

            # -- Losses
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'Mean {key} loss:':>{pad}} {value:.4f}\n"""
            # -- Rewards
            if rnd_active_for_log:
                log_string += (
                    f"""{'Mean extrinsic reward:':>{pad}} {statistics.mean(locs['erewbuffer']):.2f}\n"""
                    f"""{'Mean intrinsic reward:':>{pad}} {statistics.mean(locs['irewbuffer']):.2f}\n"""
                )
            log_string += f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
            # -- episode info
            log_string += f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )

            if arm_mean_std is not None:
                log_string += (
                    f"""{'Arm action noise std:':>{pad}} {arm_mean_std.item():.4f}\n"""
                )

            if wrist_mean_std is not None:
                log_string += (
                    f"""{'Wrist action noise std:':>{pad}} {wrist_mean_std.item():.4f}\n"""
                )

            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Time elapsed:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{'ETA:':>{pad}} {time.strftime(
                "%H:%M:%S",
                time.gmtime(
                    self.tot_time / (locs['it'] - locs['start_iter'] + 1)
                    * (locs['start_iter'] + locs['num_learning_iterations'] - locs['it'])
                )
            )}\n"""
        )
        print(log_string)

    def save(self, path: str, infos=None):
        # -- Save model
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "num_updates": getattr(self.alg, "num_updates", None),
            "scheduler_state_dict": (
                self.alg.scheduler.state_dict()
                if hasattr(self.alg, "scheduler") and self.alg.scheduler is not None
                else None
            ),
            "scheduler_steps": getattr(self.alg, "_scheduler_steps", None),
            "teacher_huber_schedule_state": (
                self.alg.get_teacher_huber_schedule_state()
                if hasattr(self.alg, "get_teacher_huber_schedule_state")
                else None
            ),
            "iteration": getattr(
                self.env.unwrapped,
                "iteration",
                None,
            ),
            # "dataset_q_pre_timestep": getattr(
            #     self.env.unwrapped,
            #     "dataset_q_pre_timestep",
            #     None,
            # ),
            # "obstacle_curriculum_difficulty": getattr(
            #     self.env.unwrapped,
            #     "obstacle_curriculum_difficulty",
            #     None,
            # ),
            # "fov_curriculum_progress": getattr(
            #     self.env.unwrapped,
            #     "fov_curriculum_progress",
            #     None,
            # ),
            "infos": infos,
        }
        # -- Save RND model if used
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        torch.save(saved_dict, path)

        # upload model to external logging service
        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def _restore_optimizer_group_cfgs(self, optimizer, current_group_cfgs, optimizer_name: str = "optimizer"):
        """Restore optimizer param_group hyperparameters from current cfg-created optimizer.

        This keeps loaded optimizer state, e.g. Adam exp_avg / exp_avg_sq, but
        overwrites param_group settings such as lr, betas, eps, weight_decay,
        amsgrad, maximize, foreach, capturable, differentiable, fused, etc.
        """
        if len(optimizer.param_groups) != len(current_group_cfgs):
            print(
                f"[WARNING] Cannot override {optimizer_name} cfg after resume: "
                f"checkpoint has {len(optimizer.param_groups)} param_groups, "
                f"current cfg has {len(current_group_cfgs)} param_groups."
            )
            return

        for group, cfg_group in zip(optimizer.param_groups, current_group_cfgs):
            group.update(cfg_group)

        print(f"[INFO] Restored {optimizer_name} param_group hyperparameters from current cfg:")
        for i, group in enumerate(optimizer.param_groups):
            shown = {
                k: group[k]
                for k in ("lr", "betas", "eps", "weight_decay", "amsgrad")
                if k in group
            }
            print(f"  {optimizer_name}.param_groups[{i}] = {shown}")

    def load(
        self,
        path: str,
        load_optimizer: bool = True,
        map_location: str | None = None,
        override_optimizer_cfg: bool = False,
    ):
        # if map_location is None:
        #     map_location = "cpu"
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)

        # Save optimizer param_group hyperparameters created from the current cfg.
        # These will be restored after loading checkpoint optimizer state.
        # This keeps Adam moments/state from checkpoint but uses current cfg values
        # for lr, betas, eps, weight_decay, etc.
        current_optimizer_group_cfgs = None
        current_rnd_optimizer_group_cfgs = None
        if override_optimizer_cfg:
            current_optimizer_group_cfgs = [
                {k: v for k, v in group.items() if k != "params"}
                for group in self.alg.optimizer.param_groups
            ]
            if hasattr(self.alg, "rnd") and self.alg.rnd and hasattr(self.alg, "rnd_optimizer"):
                current_rnd_optimizer_group_cfgs = [
                    {k: v for k, v in group.items() if k != "params"}
                    for group in self.alg.rnd_optimizer.param_groups
                ]

        # -- Load model
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        # -- Load RND model if used
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
        # -- load optimizer if used
        if load_optimizer and resumed_training:
            # -- algorithm optimizer
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])

            if override_optimizer_cfg and current_optimizer_group_cfgs is not None:
                self._restore_optimizer_group_cfgs(
                    self.alg.optimizer,
                    current_optimizer_group_cfgs,
                    optimizer_name="optimizer",
                )

            # -- RND optimizer if used
            if hasattr(self.alg, "rnd") and self.alg.rnd:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])

                if override_optimizer_cfg and current_rnd_optimizer_group_cfgs is not None:
                    self._restore_optimizer_group_cfgs(
                        self.alg.rnd_optimizer,
                        current_rnd_optimizer_group_cfgs,
                        optimizer_name="rnd_optimizer",
                    )

        # -- load current learning iteration
        if resumed_training:
            # Resume from NEXT iteration to avoid re-running same iteration
            self.current_learning_iteration = int(loaded_dict["iter"]) + 1

        # IMPORTANT: restore num_updates (fix actor-freeze inconsistency)
        if resumed_training and "num_updates" in loaded_dict and loaded_dict["num_updates"] is not None:
            if hasattr(self.alg, "num_updates"):
                self.alg.num_updates = loaded_dict["num_updates"]

        # HARD GUARD: never re-trigger actor freeze after resume
        if resumed_training and hasattr(self.alg, "freeze_actor_iterations"):
            if hasattr(self.alg, "num_updates"):
                self.alg.num_updates = max(self.alg.num_updates, self.alg.freeze_actor_iterations)

        # Restore optional scheduler state for distillation cosine LR resume.
        if resumed_training and loaded_dict.get("scheduler_state_dict", None) is not None:
            if hasattr(self.alg, "scheduler") and self.alg.scheduler is not None:
                self.alg.scheduler.load_state_dict(loaded_dict["scheduler_state_dict"])

        if resumed_training and loaded_dict.get("scheduler_steps", None) is not None:
            if hasattr(self.alg, "_scheduler_steps"):
                self.alg._scheduler_steps = loaded_dict["scheduler_steps"]

        # if resumed_training and loaded_dict.get("teacher_huber_schedule_state", None) is not None:
        #     if hasattr(self.alg, "load_teacher_huber_schedule_state"):
        #         self.alg.load_teacher_huber_schedule_state(
        #             loaded_dict["teacher_huber_schedule_state"]
        #         )
        # Do NOT restore teacher Huber schedule from checkpoint.
        # Keep the schedule created from the current cfg so that changing
        # teacher_huber_loss_weight_start/end, huber_loss_start_iters,
        # huber_loss_saturate_iters, or teacher_huber_loss_weight_schedule
        # before resume actually takes effect.
        if resumed_training and hasattr(self.alg, "teacher_huber_loss_weight"):
            self.alg.teacher_huber_loss_weight = self.alg._get_teacher_huber_loss_weight()

        restored_iteration_raw = loaded_dict.get(
            "iteration",
            None,
        )

        if resumed_training and restored_iteration_raw is not None:
            restored_iteration = int(self.current_learning_iteration)

            self.env.unwrapped.iteration = restored_iteration
            # curriculum_cfg = getattr(self.env.unwrapped.cfg, "curriculum", None)

        # infos = loaded_dict.get("infos", None)
        # del loaded_dict
        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()
        #     torch.cuda.empty_cache()
        # return infos
        return loaded_dict.get("infos", None)

    def init_policy_from_distillation(self, path: str):
        loaded_dict = torch.load(path, weights_only=False, map_location="cpu")
        model_state_dict = loaded_dict["model_state_dict"]

        model_state_dict = {k: v.cpu() if torch.is_tensor(v) else v for k, v in model_state_dict.items()}
        self.alg.policy.load_state_dict(model_state_dict, strict=True)
        return loaded_dict.get("infos", None)

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.policy.to(device)
        return self.alg.policy.act_inference

    def train_mode(self):
        # -- PPO
        self.alg.policy.train()
        # -- RND
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            self.alg.rnd.train()

    def eval_mode(self):
        # -- PPO
        self.alg.policy.eval()
        # -- RND
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            self.alg.rnd.eval()

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)

    def print_tensor_memory_snapshot(self, iteration: int = -1):
        """Print all currently allocated tensors with their shapes and memory usage.

        Args:
            iteration: Current training iteration for logging purposes

        Returns:
            List of tensor info dictionaries sorted by memory usage
        """
        # Force garbage collection first
        gc.collect()
        torch.cuda.empty_cache()

        # Get all tensor objects
        tensor_info = []
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    tensor = obj if torch.is_tensor(obj) else obj.data
                    if tensor.is_cuda:
                        size_bytes = tensor.element_size() * tensor.nelement()
                        # Try to get variable name (this is approximate)
                        name = None
                        for referrer in gc.get_referrers(obj):
                            if isinstance(referrer, dict):
                                for k, v in referrer.items():
                                    if v is obj and isinstance(k, str):
                                        name = k
                                        break

                        tensor_info.append({
                            'shape': tuple(tensor.shape),
                            'dtype': str(tensor.dtype),
                            'device': str(tensor.device),
                            'size_mb': size_bytes / 1024 / 1024,
                            'grad': tensor.requires_grad,
                            'name': name or 'unknown',
                            'numel': tensor.numel(),
                        })
            except:
                pass

        # Sort by memory usage
        tensor_info.sort(key=lambda x: x['size_mb'], reverse=True)

        # Print summary
        print(f"\n{'='*60}")
        print(f"Memory Snapshot - Iteration {iteration}")
        print(f"{'='*60}")
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated()/1e9:.3f} GB")
        print(f"GPU Memory Reserved: {torch.cuda.memory_reserved()/1e9:.3f} GB")
        print(f"Total tensors tracked: {len(tensor_info)}")

        # Print top 20 largest tensors
        print(f"\nTop 20 Largest Tensors:")
        print(f"{'#':<3} {'Shape':<30} {'Size (MB)':<12} {'Dtype':<12} {'Grad':<6} {'Name':<20}")
        print("-" * 95)

        total_mb = 0
        for i, info in enumerate(tensor_info[:20], 1):
            shape_str = str(info['shape'])[:28]
            name_str = info['name'][:18]
            print(f"{i:<3} {shape_str:<30} {info['size_mb']:>10.2f}  {info['dtype']:<12} {str(info['grad']):<6} {name_str:<20}")
            total_mb += info['size_mb']

        print(f"\nTotal memory in top 20: {total_mb:.2f} MB ({total_mb/1024:.3f} GB)")

        # Group by shape to find duplicates
        shape_groups = {}
        for info in tensor_info:
            shape = info['shape']
            if shape not in shape_groups:
                shape_groups[shape] = {'count': 0, 'total_mb': 0}
            shape_groups[shape]['count'] += 1
            shape_groups[shape]['total_mb'] += info['size_mb']

        # Print shapes with multiple instances
        duplicates = [(shape, data) for shape, data in shape_groups.items() if data['count'] > 1]
        if duplicates:
            duplicates.sort(key=lambda x: x[1]['total_mb'], reverse=True)
            print(f"\nDuplicate tensor shapes (same shape appearing multiple times):")
            print(f"{'Shape':<30} {'Count':<8} {'Total MB':<12}")
            print("-" * 50)
            for shape, data in duplicates[:10]:
                shape_str = str(shape)[:28]
                print(f"{shape_str:<30} {data['count']:<8} {data['total_mb']:>10.2f}")

        return tensor_info

    def save_memory_snapshot(self, iteration: int, save_dir: str = None):
        """Save detailed memory snapshot with tensor info to file.

        Args:
            iteration: Current training iteration
            save_dir: Directory to save snapshot (defaults to log_dir/memory_snapshots)
        """
        if save_dir is None:
            save_dir = os.path.join(self.log_dir, "memory_snapshots") if self.log_dir else "./memory_snapshots"

        os.makedirs(save_dir, exist_ok=True)

        # Get tensor info
        tensor_info = self.print_tensor_memory_snapshot(iteration)

        # Save detailed snapshot
        snapshot = {
            'iteration': iteration,
            'timestamp': time.time(),
            'gpu_allocated_gb': torch.cuda.memory_allocated() / 1e9,
            'gpu_reserved_gb': torch.cuda.memory_reserved() / 1e9,
            'num_tensors': len(tensor_info),
            'tensors': tensor_info[:100],  # Save top 100 tensors
            'cuda_memory_summary': torch.cuda.memory_summary() if torch.cuda.is_available() else None,
        }

        # Save as pickle for full data
        pickle_path = os.path.join(save_dir, f"memory_snapshot_iter_{iteration}.pkl")
        with open(pickle_path, 'wb') as f:
            pickle.dump(snapshot, f)

        # Save summary as JSON for easy reading
        json_snapshot = {
            'iteration': iteration,
            'timestamp': snapshot['timestamp'],
            'gpu_allocated_gb': snapshot['gpu_allocated_gb'],
            'gpu_reserved_gb': snapshot['gpu_reserved_gb'],
            'num_tensors': snapshot['num_tensors'],
            'top_10_tensors': [
                {
                    'shape': t['shape'],
                    'size_mb': t['size_mb'],
                    'dtype': t['dtype'],
                    'requires_grad': t['grad']
                }
                for t in tensor_info[:10]
            ]
        }

        json_path = os.path.join(save_dir, f"memory_snapshot_iter_{iteration}.json")
        with open(json_path, 'w') as f:
            json.dump(json_snapshot, f, indent=2)

        # Also save PyTorch's memory snapshot if available
        if torch.cuda.is_available():
            try:
                # Enable memory history temporarily
                torch.cuda.memory._record_memory_history(max_entries=100000)

                # Take snapshot
                torch_snapshot_path = os.path.join(save_dir, f"torch_snapshot_iter_{iteration}.pickle")
                torch.cuda.memory._dump_snapshot(torch_snapshot_path)

                # Disable to avoid overhead
                torch.cuda.memory._record_memory_history(enabled=None)

                print(f"Saved memory snapshots to {save_dir}/")
            except Exception as e:
                print(f"Warning: Could not save PyTorch memory snapshot: {e}")

    """
    Helper functions.
    """

    def _configure_multi_gpu(self):
        """Configure multi-gpu training."""
        # check if distributed training is enabled
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        # if not distributed training, set local and global rank to 0 and return
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        # get rank and world size
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        # make a configuration dictionary
        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,  # rank of the main process
            "local_rank": self.gpu_local_rank,  # rank of the current process
            "world_size": self.gpu_world_size,  # total number of processes
        }

        # check if user has device specified for local rank
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(
                f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'."
            )
        # validate multi-gpu configuration
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(
                f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(
                f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )

        # initialize torch distributed
        torch.distributed.init_process_group(backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size)
        # set device to the local rank
        torch.cuda.set_device(self.gpu_local_rank)

    def _construct_algorithm(self, obs) -> PPO:
        """Construct the actor-critic algorithm."""
        # resolve RND config
        self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)

        # resolve symmetry config
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)

        # resolve deprecated normalization config
        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn(
                "The `empirical_normalization` parameter is deprecated. Please set `actor_obs_normalization` and "
                "`critic_obs_normalization` as part of the `policy` configuration instead.",
                DeprecationWarning,
            )
            if self.policy_cfg.get("actor_obs_normalization") is None:
                self.policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
            if self.policy_cfg.get("critic_obs_normalization") is None:
                self.policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

        # initialize the actor-critic
        actor_critic_class = eval(self.policy_cfg.pop("class_name"))
        actor_critic: ActorCritic = actor_critic_class(
            obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        # initialize the algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))
        alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg)

        # initialize the storage
        alg.init_storage(
            "rl",
            self.env.num_envs,
            self.num_steps_per_env,
            obs,
            [self.env.num_actions],
        )

        return alg

    def _prepare_logging_writer(self):
        """Prepares the logging writers."""
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")
            
    def _collect_observation_data(self, obs, iteration, step=None):
        """Collect critic observations for PointNet dataset."""
        # Manually set parameters
        collect_enabled = False
        collection_interval = 25  # Collect every 10 iterations
        step_interval = 4       # Collect every 50 steps within iteration

        if not collect_enabled or iteration % collection_interval != 5:
            return

        if step is not None and step % step_interval != 0:
            return

        # Create directory
        dataset_dir = "/workspace/isaaclab/source/custom/CollisionAwareReaching/source/rl_reach/rl_reach/tasks/reach/networks/datasets/raw"
        os.makedirs(dataset_dir, exist_ok=True)

        # Save to HDF5 with iteration and step
        if step is not None:
            filepath = os.path.join(dataset_dir, f"critic_obs_iter_{iteration:06d}_step_{step:04d}.h5")
        else:
            filepath = os.path.join(dataset_dir, f"critic_obs_iter_{iteration:06d}.h5")

        with h5py.File(filepath, 'w') as f:
            # Save all critic observations
            for key, tensor in obs['critic'].items():
                f.create_dataset(key, data=tensor.cpu().numpy(), compression='gzip')

            # Save metadata
            f.attrs['iteration'] = iteration
            f.attrs['num_envs'] = self.env.num_envs
            if step is not None:
                f.attrs['step'] = step
