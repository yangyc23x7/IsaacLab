# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn.functional as F
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from .dextrah_kuka_allegro_env_cfg import DextrahKukaAllegroEnvCfg


class DextrahKukaAllegroEnv(DirectRLEnv):
    cfg: DextrahKukaAllegroEnvCfg

    def __init__(self, cfg: DextrahKukaAllegroEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.cfg.robot_scene_cfg.resolve(self.scene)
        self.object_goal = torch.tensor(self.cfg.object_goal, device=self.device).repeat((self.num_envs, 1))
        self.object_goal += self.scene.env_origins
        self.curled_q = torch.tensor(self.cfg.curled_q, device=self.device).repeat(self.num_envs, 1).contiguous()
        self.joint_pos_action = self.cfg.joint_pos_action_cfg.class_type(self.cfg.joint_pos_action_cfg, self)
        self.joint_vel_action = self.cfg.joint_vel_action_cfg.class_type(self.cfg.joint_vel_action_cfg, self)
        # Track success statistics
        center = self.cfg.objects_cfg.init_state.pos
        self.oob_limits = torch.tensor([
            [center[0] - self.cfg.obj_spawn_width[0] / 2., center[1] - self.cfg.obj_spawn_width[1] / 2., 0.2],
            [center[0] + self.cfg.obj_spawn_width[0] / 2., center[1] + self.cfg.obj_spawn_width[1] / 2., float('inf')]
        ], device=self.device)

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in ["hand_to_object", "object_to_goal", "finger_curl_reg", "lift_reward"]
        }

    def _setup_scene(self):
        # add robot, objects
        self.robot = Articulation(self.cfg.robot_cfg)
        self.table = RigidObject(self.cfg.table_cfg)
        self.object = RigidObject(self.cfg.objects_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # add articultion to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["table"] = self.table
        self.scene.rigid_objects["object"] = self.object
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        num_unique_objects = len(self.object.cfg.spawn.assets_cfg)
        num_teacher_observations = 147 + num_unique_objects
        self.cfg.state_space = 179 + num_unique_objects
        self.cfg.observation_space = num_teacher_observations

        self.multi_object_idx = torch.remainder(torch.arange(self.num_envs), num_unique_objects).to(self.device)
        self.multi_object_idx_onehot = F.one_hot(self.multi_object_idx, num_classes=num_unique_objects).float()


    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Find the current global minimum adr increment
        self._actions = actions.clone()
        self.joint_pos_action.process_actions(actions[:, :23])
        self.joint_vel_action.process_actions(actions[:, 23:])

    def _apply_action(self) -> None:
        self.joint_pos_action.apply_actions()
        self.joint_vel_action.apply_actions()

    def _get_observations(self) -> dict:
        joint_pos = self.robot.data.joint_pos[:, self.cfg.robot_scene_cfg.joint_ids]
        joint_vel = self.robot.data.joint_vel[:, self.cfg.robot_scene_cfg.joint_ids]
        hand_pos = self.robot.data.body_pos_w[:, self.cfg.robot_scene_cfg.body_ids].view(self.num_envs, -1)
        hand_pos -= self.scene.env_origins.repeat((1, len(self.cfg.robot_scene_cfg.body_ids)))
        hand_vel = self.robot.data.body_vel_w[:, self.cfg.robot_scene_cfg.body_ids].view(self.num_envs, -1)
        object_pos = self.object.data.root_pos_w - self.scene.env_origins
        object_rot = self.object.data.root_quat_w
        hand_forces = self.robot.root_physx_view.get_link_incoming_joint_force()[:, self.cfg.robot_scene_cfg.body_ids]
        measured_joint_torques = self.robot.root_physx_view.get_dof_projected_joint_forces()

        teacher_policy_obs = torch.cat(
            (
                joint_pos,  # 0:23
                joint_vel,  # 23:46
                hand_pos,  # 46:61
                hand_vel,  # 61:91
                object_pos,  # 91:94
                object_rot,  # 94:98
                self.object_goal - self.scene.env_origins,  # 98:101
                self.multi_object_idx_onehot,  # 101:253
                self._actions,  # 254:300
            ), dim=-1,
        )

        critic_obs = torch.cat(
            (
                joint_pos,  # 0:23
                joint_vel,  # 23:46
                hand_pos,  # 46:61
                hand_vel,  # 61:76
                hand_forces.view(self.num_envs, -1),
                measured_joint_torques,
                object_pos,
                object_rot,
                self.object.data.root_vel_w,
                self.object_goal - self.scene.env_origins,
                self.multi_object_idx_onehot,
                self._actions,
            ), dim=-1,
        )

        observations = {"policy": teacher_policy_obs, "critic": critic_obs}

        return observations

    def _get_rewards(self) -> torch.Tensor:
        hand_pos = self.robot.data.body_pos_w[:, self.cfg.robot_scene_cfg.body_ids]
        object_pos = self.object.data.root_pos_w
        joint_pos = self.robot.data.joint_pos[:, self.cfg.robot_scene_cfg.joint_ids]

        object_to_goal_pos_error = torch.norm(object_pos - self.object_goal, dim=-1)
        object_vertical_error = torch.abs(self.object_goal[:, 2] - object_pos[:, 2])
        hand_to_object_pos_error = torch.norm(hand_pos - object_pos[:, None, :], dim=-1).max(dim=-1).values
        in_success_region = torch.norm(object_pos - self.object_goal, dim=-1) < self.cfg.object_goal_tol

        hand_to_object_rew = torch.exp(-self.cfg.hand_to_object_sharpness * hand_to_object_pos_error)
        object_to_goal_rew = torch.exp(-self.cfg.object_to_goal_sharpness * object_to_goal_pos_error)
        finger_curl_reg = torch.sum(torch.square(joint_pos[:, 7:] - self.curled_q), dim=-1)
        lift_rew = torch.exp(-self.cfg.lift_sharpness * object_vertical_error)

        self.extras["in_success_region"] = in_success_region.float().mean()

        rewards = {
            "hand_to_object" : self.cfg.hand_to_object_weight * hand_to_object_rew * self.step_dt,
            "object_to_goal" : self.cfg.object_to_goal_weight * object_to_goal_rew * self.step_dt,
            "finger_curl_reg" : self.cfg.finger_curl_reg_weight * finger_curl_reg * self.step_dt,
            "lift_reward" : self.cfg.lift_weight * lift_rew * self.step_dt
        }
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return torch.sum(torch.stack(list(rewards.values())), dim=0)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        object_pos = self.object.data.root_pos_w - self.scene.env_origins
        outside_bounds = ((object_pos < self.oob_limits[0]) | (object_pos > self.oob_limits[1])).any(dim=1)
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return outside_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        super()._reset_idx(env_ids)
        # Reset object state
        object_start_state = self.object.data.default_root_state[env_ids].clone()
        object_start_state[:, :3] += self.scene.env_origins[env_ids]
        self.object.write_root_state_to_sim(object_start_state, env_ids)

        default_joint_pos = self.robot.data.default_joint_pos[env_ids[:, None], self.cfg.robot_scene_cfg.joint_ids]
        default_joint_vel = self.robot.data.default_joint_vel[env_ids[:, None], self.cfg.robot_scene_cfg.joint_ids]
        self.robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids, joint_ids=self.cfg.robot_scene_cfg.joint_ids)
        self.robot.set_joint_position_target(default_joint_pos, env_ids=env_ids, joint_ids=self.cfg.robot_scene_cfg.joint_ids)
        self.robot.set_joint_velocity_target(default_joint_vel, env_ids=env_ids, joint_ids=self.cfg.robot_scene_cfg.joint_ids)

        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
