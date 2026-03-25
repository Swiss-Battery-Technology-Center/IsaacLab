# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for pose tracking."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import (
    axis_angle_relative_from_quat,
    combine_frame_transforms,
    compute_pose_error,
    normalize,
    quat_from_euler_xyz,
    quat_from_relative_axis_angle,
    quat_from_rot6d,
    quat_unique,
    rot6d_from_quat,
)


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import UniformPoseCommandCfg, UniformPoseEncodedCommandCfg


class UniformPoseCommand(CommandTerm):
    """Command generator for generating pose commands uniformly.

    The command generator generates poses by sampling positions uniformly within specified
    regions in cartesian space. For orientation, it samples uniformly the euler angles
    (roll-pitch-yaw) and converts them into quaternion representation (w, x, y, z).

    The position and orientation commands are generated in the base frame of the robot, and not the
    simulation world frame. This means that users need to handle the transformation from the
    base frame to the simulation world frame themselves.

    .. caution::

        Sampling orientations uniformly is not strictly the same as sampling euler angles uniformly.
        This is because rotations are defined by 3D non-Euclidean space, and the mapping
        from euler angles to rotations is not one-to-one.

    """

    cfg: UniformPoseCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: UniformPoseCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # extract the robot and body index for which the command is generated
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]

        # create buffers
        # -- commands: (x, y, z, qw, qx, qy, qz) in root frame
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0
        self.pose_command_w = torch.zeros_like(self.pose_command_b)
        # -- metrics
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "UniformPoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired pose command. Shape is (num_envs, 7).

        The first three elements correspond to the position, followed by the quaternion orientation in (w, x, y, z).
        """
        return self.pose_command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # transform command from base frame to simulation world frame
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
        )
        # compute the error
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.robot.data.body_pos_w[:, self.body_idx],
            self.robot.data.body_quat_w[:, self.body_idx],
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        # sample new pose targets
        # -- position
        r = torch.empty(len(env_ids), device=self.device)
        self.pose_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x)
        self.pose_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.pos_y)
        self.pose_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.pos_z)
        # -- orientation
        euler_angles = torch.zeros_like(self.pose_command_b[env_ids, :3])
        euler_angles[:, 0].uniform_(*self.cfg.ranges.roll)
        euler_angles[:, 1].uniform_(*self.cfg.ranges.pitch)
        euler_angles[:, 2].uniform_(*self.cfg.ranges.yaw)
        quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        # make sure the quaternion has real part as positive
        self.pose_command_b[env_ids, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                # -- goal pose
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                # -- current body pose
                self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
            # set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # update the markers
        # -- goal pose
        self.goal_pose_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:])
        # -- current body pose
        body_link_pose_w = self.robot.data.body_link_pose_w[:, self.body_idx]
        self.current_pose_visualizer.visualize(body_link_pose_w[:, :3], body_link_pose_w[:, 3:7])


class UniformPoseEncodedCommand(UniformPoseCommand):
    """Uniform pose command with configurable exposed orientation representation.

    Compatibility:
      - still exposes .command
      - command always begins with xyz position
      - metrics names remain unchanged
      - debug vis always receives quaternion pose
    """

    cfg: UniformPoseEncodedCommandCfg

    def __init__(self, cfg: UniformPoseEncodedCommandCfg, env):
        super().__init__(cfg, env)

        if self.cfg.orientation_representation == "quat":
            # Reuse parent buffers exactly for full backward compatibility.
            self.pose_command_b_quat = self.pose_command_b
            self.pose_command_w_quat = self.pose_command_w
        else:
            # Canonical internal command: always pos + quat
            self.pose_command_b_quat = torch.zeros(self.num_envs, 7, device=self.device)
            self.pose_command_b_quat[:, 3] = 1.0

            self.pose_command_w_quat = torch.zeros_like(self.pose_command_b_quat)
            self.pose_command_w_quat[:, 3] = 1.0

            # Replace public exposed command buffer with representation-dependent layout
            if self.cfg.orientation_representation == "axis_angle":
                cmd_dim = 6
            elif self.cfg.orientation_representation == "rot6d":
                cmd_dim = 9
            else:
                raise ValueError(
                    f"Unsupported orientation_representation: {self.cfg.orientation_representation}"
                )

            self.pose_command_b = torch.zeros(self.num_envs, cmd_dim, device=self.device)
            self._update_command_representation()

    def __str__(self) -> str:
        msg = "UniformPoseEncodedCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tOrientation representation: {self.cfg.orientation_representation}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        if self.cfg.orientation_representation == "axis_angle":
            msg += f"\tAxis-angle reference quat: {self.cfg.axis_angle_reference_quat}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """Desired pose command.

        Layout:
          - quat:       (x, y, z, qw, qx, qy, qz)
          - axis_angle: (x, y, z, ax, ay, az)
          - rot6d:      (x, y, z, r1, r2, r3, r4, r5, r6)
        """
        return self.pose_command_b

    def get_command_quat_b(self) -> torch.Tensor:
        """Canonical internal base-frame pose command as pos + quat."""
        return self.pose_command_b_quat

    def get_command_quat_w(self) -> torch.Tensor:
        """Canonical internal world-frame pose command as pos + quat."""
        return self.pose_command_w_quat

    def _axis_angle_reference_quat_tensor(self, batch_size: int, device: torch.device) -> torch.Tensor:
        ref = torch.tensor(self.cfg.axis_angle_reference_quat, device=device, dtype=torch.float32)
        return ref.unsqueeze(0).repeat(batch_size, 1)

    def _update_command_representation(self, env_ids: Sequence[int] | None = None):
        """Convert canonical quaternion command into configured exposed representation."""
        if env_ids is None:
            pos = self.pose_command_b_quat[:, :3]
            quat = self.pose_command_b_quat[:, 3:]

            self.pose_command_b[:, :3] = pos

            if self.cfg.orientation_representation == "quat":
                self.pose_command_b[:, 3:7] = quat

            elif self.cfg.orientation_representation == "axis_angle":
                ref_quat = self._axis_angle_reference_quat_tensor(quat.shape[0], quat.device).to(quat.dtype)
                self.pose_command_b[:, 3:6] = axis_angle_relative_from_quat(quat, ref_quat)

            elif self.cfg.orientation_representation == "rot6d":
                self.pose_command_b[:, 3:9] = rot6d_from_quat(quat)

            else:
                raise ValueError(
                    f"Unsupported orientation_representation: {self.cfg.orientation_representation}"
                )

        else:
            pos = self.pose_command_b_quat[env_ids, :3]
            quat = self.pose_command_b_quat[env_ids, 3:]

            self.pose_command_b[env_ids, :3] = pos

            if self.cfg.orientation_representation == "quat":
                self.pose_command_b[env_ids, 3:7] = quat

            elif self.cfg.orientation_representation == "axis_angle":
                ref_quat = self._axis_angle_reference_quat_tensor(quat.shape[0], quat.device).to(quat.dtype)
                self.pose_command_b[env_ids, 3:6] = axis_angle_relative_from_quat(quat, ref_quat)

            elif self.cfg.orientation_representation == "rot6d":
                self.pose_command_b[env_ids, 3:9] = rot6d_from_quat(quat)

            else:
                raise ValueError(
                    f"Unsupported orientation_representation: {self.cfg.orientation_representation}"
                )
    def decode_orientation_to_quat_b(self, orientation: torch.Tensor) -> torch.Tensor:
        """Decode orientation-only tensor to quaternion."""
        ori_dim = orientation.shape[-1]

        if ori_dim == 4:
            return normalize(orientation)
        elif ori_dim == 3:
            ref_quat = self._axis_angle_reference_quat_tensor(orientation.shape[0], orientation.device).to(orientation.dtype)
            return quat_from_relative_axis_angle(orientation, ref_quat)
        elif ori_dim == 6:
            return quat_from_rot6d(orientation)
        else:
            raise ValueError(f"Unsupported orientation dimension: {ori_dim}")

    def decode_command_to_quat_b(self, command: torch.Tensor | None = None) -> torch.Tensor:
        """Decode full command tensor to quaternion in base frame."""
        if command is None:
            command = self.command
        return self.decode_orientation_to_quat_b(command[:, 3:])

    def _resample_command(self, env_ids: Sequence[int]):
        # -- position (same as original)
        r = torch.empty(len(env_ids), device=self.device)
        self.pose_command_b_quat[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x)
        self.pose_command_b_quat[env_ids, 1] = r.uniform_(*self.cfg.ranges.pos_y)
        self.pose_command_b_quat[env_ids, 2] = r.uniform_(*self.cfg.ranges.pos_z)

        # -- orientation (sampled canonically as quat)
        euler_angles = torch.zeros(len(env_ids), 3, device=self.device)
        euler_angles[:, 0].uniform_(*self.cfg.ranges.roll)
        euler_angles[:, 1].uniform_(*self.cfg.ranges.pitch)
        euler_angles[:, 2].uniform_(*self.cfg.ranges.yaw)

        quat = quat_from_euler_xyz(
            euler_angles[:, 0],
            euler_angles[:, 1],
            euler_angles[:, 2],
        )
        quat = quat_unique(quat) if self.cfg.make_quat_unique else normalize(quat)
        self.pose_command_b_quat[env_ids, 3:] = quat

        # update public representation buffer
        self._update_command_representation(env_ids)
        if self.cfg.orientation_representation == "quat":
            max_diff = (self.pose_command_b[env_ids] - self.pose_command_b_quat[env_ids]).abs().max()
            if max_diff > 1e-6:
                raise RuntimeError(f"quat command mismatch after resample: {max_diff.item()}")

        elif self.cfg.orientation_representation == "rot6d":
            q_ref = normalize(self.pose_command_b_quat[env_ids, 3:7])
            d6 = self.pose_command_b[env_ids, 3:9]
            q_dec = normalize(quat_from_rot6d(d6))

            abs_dot = torch.abs(torch.sum(q_ref * q_dec, dim=-1))
            min_abs_dot = abs_dot.min()

            if min_abs_dot < 1.0 - 1e-5:
                raise RuntimeError(
                    f"rot6d roundtrip mismatch after resample: "
                    f"min_abs_dot={min_abs_dot.item()}, "
                    f"mean_abs_dot={abs_dot.mean().item()}"
                )
    def _update_command(self):
        # piecewise constant between resamples
        pass

    def _update_metrics(self):
        """Compute metrics using canonical quaternion command."""
        self.pose_command_w_quat[:, :3], self.pose_command_w_quat[:, 3:] = combine_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.pose_command_b_quat[:, :3],
            self.pose_command_b_quat[:, 3:],
        )

        pos_error, rot_error = compute_pose_error(
            self.pose_command_w_quat[:, :3],
            self.pose_command_w_quat[:, 3:],
            self.robot.data.body_pos_w[:, self.body_idx],
            self.robot.data.body_quat_w[:, self.body_idx],
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.pose_command_w_quat[:, :3], self.pose_command_w_quat[:, 3:] = combine_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.pose_command_b_quat[:, :3],
            self.pose_command_b_quat[:, 3:],
        )

        self.goal_pose_visualizer.visualize(
            self.pose_command_w_quat[:, :3],
            self.pose_command_w_quat[:, 3:],
        )

        body_link_pose_w = self.robot.data.body_link_pose_w[:, self.body_idx]
        self.current_pose_visualizer.visualize(body_link_pose_w[:, :3], body_link_pose_w[:, 3:7])

