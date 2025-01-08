# -*- coding: utf-8 -*-
# @Auther   : Zheng SUN (ZainZh)
# @Time     : 2025/1/7
# @Address  : clover Lab @ CUHK
# @FileName : curi_sim.py

# @Description : TODO
from isaacgym import gymapi
import torch
from utilis.common import print_info
from base_sim import RobotSim
from typing import List
import numpy as np
import math


class CURISim(RobotSim):
    def __init__(self, config="curi"):
        super().__init__(config)

    def interact_with_curi(self):
        while not self.gym.query_viewer_has_closed(self.viewer):
            # Step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            # refresh tensors
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_net_contact_force_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_jacobian_tensors(self.sim)

            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)
            self.gym.sync_frame_time(self.sim)

    def run_traj_multi_rigid_bodies(
        self,
        traj: List,
        attr_rbs: List = None,
        attr_types=None,
        update_freq=0.001,
        verbose=True,
        index_list=None,
        recursive_play=False,
    ):
        """
        Set multiple attractors to let the robot run the trajectory with multiple rigid bodies.

        :param traj: a list of trajectories, each trajectory is a numpy array of shape (N, 7)
        :param attr_rbs: [list], e.g. ["panda_left_hand", "panda_right_hand"]
        :param attr_types: [list], e.g. [gymapi.AXIS_ALL, gymapi.AXIS_ROTATION, gymapi.AXIS_TRANSLATION]
        :param root_state: the root state of the robot
        :param update_freq: the frequency of updating the robot pose
        :param verbose: if True, visualize the attractor spheres
        :param index_list:
        :param recursive_play:
        :return:
        """
        from isaacgym import gymtorch

        assert (
            isinstance(traj, list) and len(traj) > 0
        ), "The trajectory should be a list of numpy arrays"
        print_info("Execute multi rigid bodies trajectory")

        self.gym.prepare_sim(self.sim)
        self.monitor_rigid_body_states()
        self.monitor_actor_root_states()
        self.monitor_dof_states()

        self.robot_root_states = self.root_states.view(self.num_envs, 1, -1)[..., 0, :]

        # Create the attractor
        attr_rbs, attr_handles, axes_geoms, sphere_geoms = self._setup_attractors(
            traj, attr_rbs, attr_types, verbose=verbose
        )

        # Time to wait in seconds before moving robot
        next_update_time = 1
        index = 0
        dof_states = torch.zeros(
            (traj[0].shape[0], self.dof_states.shape[0], self.dof_states.shape[1]),
            dtype=torch.float32,
        )
        save_dof_states = True
        while not self.gym.query_viewer_has_closed(self.viewer):
            # Every 0.01 seconds the pose of the attractor is updated
            t = self.gym.get_sim_time(self.sim)

            if t >= next_update_time:
                self.gym.refresh_rigid_body_state_tensor(self.sim)
                self.gym.refresh_actor_root_state_tensor(self.sim)
                self.gym.refresh_dof_state_tensor(self.sim)

                if save_dof_states:
                    dof_states[index] = self.dof_states.clone()

                self.gym.clear_lines(self.viewer)
                for i in range(len(attr_rbs)):
                    self.update_robot(
                        traj[i],
                        attr_handles[i],
                        axes_geoms[i],
                        sphere_geoms[i],
                        index,
                        verbose,
                        index_list=index_list,
                    )

                next_update_time += update_freq
                index += 1
                if index >= len(traj[i]):
                    if recursive_play:
                        index = 0
                    else:
                        break  # stop the simulation

            # Step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            # Step rendering
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)
            self.gym.sync_frame_time(self.sim)

        print("Done")
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

        return dof_states

    def _init_attractor(self, attracted_rigid_body, attr_type=None, verbose=True):
        """
        Initialize the attractor for tracking the trajectory using the embedded Isaac Gym PID controller

        :param attracted_rigid_body: the joint to be attracted
        :param attr_type: the type of the attractor
        :param verbose: if True, visualize the attractor spheres
        :return:
        """
        from isaacgym import gymapi
        from isaacgym import gymutil

        # Attractor setup
        attractor_handles = []
        attractor_properties = gymapi.AttractorProperties()
        # Make attractor in all axes
        attractor_properties.axes = attr_type
        attractor_properties.stiffness = (
            5e5
            if attr_type == gymapi.AXIS_ALL or attr_type == gymapi.AXIS_TRANSLATION
            else 5000
        )
        attractor_properties.damping = (
            5e3
            if attr_type == gymapi.AXIS_ALL or attr_type == gymapi.AXIS_TRANSLATION
            else 500
        )

        # Create helper geometry used for visualization
        # Create a wireframe axis
        axes_geom = gymutil.AxesGeometry(0.1)
        # Create a wireframe sphere
        sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
        sphere_pose = gymapi.Transform(r=sphere_rot)
        if attr_type == gymapi.AXIS_ALL:
            sphere_geom = gymutil.WireframeSphereGeometry(
                0.003, 12, 12, sphere_pose, color=(1, 0, 0)
            )
        elif attr_type == gymapi.AXIS_ROTATION:
            sphere_geom = gymutil.WireframeSphereGeometry(
                0.003, 12, 12, sphere_pose, color=(0, 1, 0)
            )
        elif attr_type == gymapi.AXIS_TRANSLATION:
            sphere_geom = gymutil.WireframeSphereGeometry(
                0.003, 12, 12, sphere_pose, color=(0, 0, 1)
            )
        else:
            sphere_geom = gymutil.WireframeSphereGeometry(
                0.003, 12, 12, sphere_pose, color=(1, 1, 1)
            )

        for i in range(len(self.envs)):
            env = self.envs[i]
            handle = self.robot_handles[i]

            body_dict = self.gym.get_actor_rigid_body_dict(env, handle)
            # beauty_print(f"get_actor_rigid_body_dict: {body_dict}")
            props = self.gym.get_actor_rigid_body_states(env, handle, gymapi.STATE_POS)
            attracted_rigid_body_handle = self.gym.find_actor_rigid_body_handle(
                env, handle, attracted_rigid_body
            )

            # Initialize the attractor
            attractor_properties.target = props["pose"][:][
                body_dict[attracted_rigid_body]
            ]
            attractor_properties.rigid_handle = attracted_rigid_body_handle

            if verbose:
                # Draw axes and sphere at attractor location
                gymutil.draw_lines(
                    axes_geom, self.gym, self.viewer, env, attractor_properties.target
                )
                gymutil.draw_lines(
                    sphere_geom, self.gym, self.viewer, env, attractor_properties.target
                )

            attractor_handle = self.gym.create_rigid_body_attractor(
                env, attractor_properties
            )
            attractor_handles.append(attractor_handle)
        return attractor_handles, axes_geom, sphere_geom

    def _setup_attractors(self, traj, attracted_links, attr_types, verbose=True):
        """
        Setup the attractors for tracking the trajectory using the embedded Isaac Gym PID controller

        :param traj: the trajectory to be tracked
        :param attracted_links: link names to be attracted, same dim as traj
        :param attr_types: the type of the attractors
        :param verbose: if True, visualize the attractor spheres
        :return:
        """
        assert isinstance(attracted_links, list), "The attracted joints should be a list"
        assert len(attracted_links) == len(
            traj
        ), "The first dimension of trajectory should equal to attr_rbs"
        assert len(attracted_links) == len(
            attr_types
        ), "The first dimension of attr_types should equal to attr_rbs"

        attractor_handles, axes_geoms, sphere_geoms = [], [], []
        for i in range(len(attracted_links)):
            attractor_handle, axes_geom, sphere_geom = self._init_attractor(
                attracted_links[i], attr_type=attr_types[i], verbose=verbose
            )
            attractor_handles.append(attractor_handle)
            axes_geoms.append(axes_geom)
            sphere_geoms.append(sphere_geom)
        return attracted_links, attractor_handles, axes_geoms, sphere_geoms


if __name__ == "__main__":
    curi_sim = CURISim()
    curi_sim.interact_with_curi()
    traj_l = np.load("../../../assets/data/LQT_LQR/taichi_2l.npy")
    traj_r = np.load("../../../assets/data/LQT_LQR/taichi_2r.npy")
    # curi_sim.run_traj_multi_rigid_bodies(
    #     traj=[traj_l, traj_r],
    #     attr_rbs=["panda_left_link7", "panda_right_link7"],
    #     attr_types=[gymapi.AXIS_ALL, gymapi.AXIS_ALL],
    #     update_freq=0.001,
    # )
