# Description: The base simulator for the urdf-centered simulation

import math
import os
from typing import List
from isaacgym import gymapi
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image as Im

import rofunc as rf
from rofunc.utils.logger.beauty_logger import beauty_print
from utilis.common import load_omega_config, print_info


class BaseSim:
    def __init__(self, config="base_sim"):
        # init_parameters
        self.robot_asset_root = None
        self.robot_asset_file = None
        self.robot_name = None
        self.robot_asset = None
        self.envs = None
        self.robot_handles = None
        self.gym = None
        self.sim_params = None
        self.physics_engine = None

        self.config = load_omega_config(config)
        self.up_axis = None
        self.asset_root = self.config["asset_root"]
        self.init_sim()
        self.init_viewer()
        self.init_plane()
        # self.init_terrain()
        self.init_light()

    def __del__(self):
        self.gym.destroy_viewer()
        self.gym.destroy_sim()

    def init_sim(self):
        self.up_axis = self.config["sim"]["up_axis"].upper()
        # Initialize gym
        self.gym = gymapi.acquire_gym()

        # Configure sim
        self.sim_params = gymapi.SimParams()
        self.sim_params.dt = self.config["sim"]["dt"]
        self.sim_params.substeps = self.config["sim"]["sub_steps"]
        self.sim_params.gravity = gymapi.Vec3(*self.config["sim"]["gravity"])
        self.sim_params.up_axis = (
            gymapi.UP_AXIS_Y if self.up_axis == "Y" else gymapi.UP_AXIS_Z
        )

        if self.config["physics_engine"] == "flex":
            self.physics_engine = gymapi.SIM_FLEX
            for flex_param in self.config.sim.flex:
                setattr(
                    self.config.sim.flex, flex_param, self.config.sim.flex[flex_param]
                )
        elif self.config["physics_engine"] == "physx":
            self.physics_engine = gymapi.SIM_PHYSX
            for physx_param in self.config.sim.physx:
                setattr(
                    self.config.sim.physx,
                    physx_param,
                    self.config.sim.physx[physx_param],
                )
        else:
            raise ValueError("The physics engine should be in [flex, physx]")

        self.sim_params.use_gpu_pipeline = self.config.sim.use_gpu_pipeline
        if self.sim_params.use_gpu_pipeline:
            beauty_print("WARNING: Forcing CPU pipeline.", type="warning")

        split_device = self.config.sim_device.split(":")
        self.device_id = int(split_device[1]) if len(split_device) > 1 else 0
        self.sim = self.gym.create_sim(
            self.device_id,
            self.config.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )

        if self.sim is None:
            beauty_print("Failed to create sim", type="warning")
            quit()

    def init_viewer(self):
        from isaacgym import gymapi

        if self.up_axis == "Y":
            cam_pos = self.config.get("cam_pos", (3.0, 2.0, 0.0))
            cam_target = self.config.get("cam_target", (0.0, 0.0, 0.0))
        elif self.up_axis == "Z":
            cam_pos = self.config.get("cam_pos", (-3.0, 0.0, 2.0))
            cam_target = self.config.get("cam_target", (1.0, 0.0, 1.0))

        # Create viewer
        camera_props = gymapi.CameraProperties()
        camera_props.horizontal_fov = self.config.get("camera_horizontal_fov", 20.0)
        camera_props.width = self.config.get("camera_width", 1920)
        camera_props.height = self.config.get("camera_height", 1080)
        camera_props.use_collision_geometry = self.config.get(
            "camera_use_collision_geometry", False
        )
        self.viewer = self.gym.create_viewer(self.sim, camera_props)
        if self.viewer is None:
            beauty_print("Failed to create viewer", type="warning")
            quit()

        # Point camera at environments
        self.gym.viewer_camera_look_at(
            self.viewer, None, gymapi.Vec3(*cam_pos), gymapi.Vec3(*cam_target)
        )

    def init_plane(self):
        from isaacgym import gymapi

        # Add ground plane
        plane_params = gymapi.PlaneParams()
        if self.up_axis == "Y":
            plane_params.normal = gymapi.Vec3(0, 1, 0)
        elif self.up_axis == "Z":
            plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

    def init_terrain(self):
        from isaacgym import gymapi
        from isaacgym.terrain_utils import (
            SubTerrain,
            convert_heightfield_to_trimesh,
            sloped_terrain,
        )

        horizontal_scale = 1  # [m]
        vertical_scale = 1  # [m]

        # def new_sub_terrain(): return SubTerrain()

        heightfield = sloped_terrain(SubTerrain(), slope=-0).height_field_raw
        vertices, triangles = convert_heightfield_to_trimesh(
            heightfield,
            horizontal_scale=horizontal_scale,
            vertical_scale=vertical_scale,
            slope_threshold=1.5,
        )
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = vertices.shape[0]
        tm_params.nb_triangles = triangles.shape[0]
        tm_params.transform.p.x = -128.0
        tm_params.transform.p.y = -128.0
        self.gym.add_triangle_mesh(
            self.sim, vertices.flatten(), triangles.flatten(), tm_params
        )

    def init_light(self):
        from isaacgym import gymapi

        l_color = gymapi.Vec3(1, 1, 1)
        l_ambient = gymapi.Vec3(0.12, 0.12, 0.12)
        l_direction = gymapi.Vec3(-1, 0, 1)
        self.gym.set_light_parameters(self.sim, 0, l_color, l_ambient, l_direction)

    def add_object(self):
        asset_files = self.config.env.object_asset.assetFiles

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = self.config.env.object_asset.fix_base_link
        asset_options.use_mesh_materials = True
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        asset_options.override_com = True
        asset_options.override_inertia = True
        asset_options.vhacd_enabled = True
        asset_options.disable_gravity = False
        asset_options.vhacd_params = gymapi.VhacdParams()
        asset_options.vhacd_params.resolution = 100000
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        init_poses = self.config.env.object_asset.init_poses
        object_names = self.config.env.object_asset.object_names

        self.object_assets = []
        for asset_file in asset_files:
            beauty_print(
                "Loading object asset {} from {}".format(asset_file, self.asset_root),
                type="info",
            )
            self.object_assets.append(
                self.gym.load_asset(
                    self.sim, self.asset_root, asset_file, asset_options
                )
            )

        self.object_idxs = []
        self.object_handles = []
        for j in range(len(self.object_assets)):
            self.object_idxs.append([])
            self.object_handles.append([])
            for i, env_ptr in enumerate(self.envs):
                object_start_pose = gymapi.Transform()
                object_start_pose.p = gymapi.Vec3(*init_poses[j][:3])
                object_start_pose.r = gymapi.Quat(*init_poses[j][3:7])

                object_handle = self.gym.create_actor(
                    env_ptr,
                    self.object_assets[j],
                    object_start_pose,
                    object_names[j],
                    i,
                    0,
                    0,
                )
                self.object_handles[j].append(object_handle)
                object_idx = self.gym.get_actor_rigid_body_index(
                    env_ptr, object_handle, 0, gymapi.DOMAIN_SIM
                )
                self.object_idxs[j].append(object_idx)
        return self.object_handles, self.object_idxs

    def add_table(self):
        from isaacgym import gymapi

        table_size = self.config.env.table.size
        init_pose = self.config.env.table.init_pose

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = self.config.env.table.fix_base_link
        self.table_asset = self.gym.create_box(
            self.sim, table_size[0], table_size[1], table_size[2], asset_options
        )

        for i, env_ptr in enumerate(self.envs):
            table_pose = gymapi.Transform()
            table_pose.p = gymapi.Vec3(*init_pose[:3])
            table_pose.r = gymapi.Quat(*init_pose[3:7])
            self.gym.create_actor(
                env_ptr, self.table_asset, table_pose, "table", i, 0, 0
            )


class RobotSim(BaseSim):
    def __init__(self, config):
        """
        Initialize the urdf-centered simulator

        :param config: config file
        """
        super().__init__(f"robots/{config}")
        self.num_envs = self.config["envs"]["numEnvs"]
        self.num_envs_per_row = self.config["envs"]["numEnvPerRow"]
        self.robot_controller = self.config["envs"]["controller_type"]
        self.collision_mode = self.config["envs"]["collision_mode"]

        self.create_env()
        self.setup_robot_dof_prop()
        if self.config["envs"]["object_asset"] is not None:
            self.add_object()
        if hasattr(self.config["envs"], "table"):
            if self.config["envs"]["table"] is not None:
                self.add_table()

    def setup_robot_dof_prop(self):
        robot_dof_props = self.gym.get_asset_dof_properties(self.robot_asset)
        robot_lower_limits = robot_dof_props["lower"]
        robot_upper_limits = robot_dof_props["upper"]
        robot_dof_props["stiffness"][:] = 300
        robot_dof_props["damping"][:] = 30
        robot_dof_props["driveMode"][:].fill(gymapi.DOF_MODE_POS)

        # # default dof states and position targets
        # robot_num_dofs = gym.get_asset_dof_count(robot_asset)
        # default_dof_pos = np.zeros(robot_num_dofs, dtype=np.float32)
        # # default_dof_pos[:] = robot_mids[:]
        #
        # default_dof_state = np.zeros(robot_num_dofs, gymapi.DofState.dtype)
        # default_dof_state["pos"] = default_dof_pos

        for env, robot_handle in zip(self.envs, self.robot_handles):
            # set dof properties
            self.gym.set_actor_dof_properties(env, robot_handle, robot_dof_props)

            # # set initial dof states
            # self.gym.set_actor_dof_states(env, robot_handle, self.default_dof_state, gymapi.STATE_ALL)
            #
            # # set initial position targets
            # self.gym.set_actor_dof_position_targets(env, robot_handle, default_dof_pos)

    def get_dof_info(self):
        # Gets number of Degree of Freedom for an actor
        dof_count = self.gym.get_actor_dof_count(self.envs[0], self.robot_handles[0])
        # maps degree of freedom names to actor-relative indices
        dof_dict = self.gym.get_actor_dof_dict(self.envs[0], self.robot_handles[0])
        # Gets forces for the actor’s degrees of freedom
        # dof_forces = self.gym.get_actor_dof_forces(self.envs[0], self.robot_handles[0])
        # Gets Frames for Degrees of Freedom of actor
        # dof_frames = self.gym.get_actor_dof_frames(self.envs[0], self.robot_handles[0])
        # Gets names of all degrees of freedom on actor
        dof_names = self.gym.get_actor_dof_names(self.envs[0], self.robot_handles[0])
        # Gets target position for the actor’s degrees of freedom.
        # dof_position_targets = self.gym.get_actor_dof_position_targets(self.envs[0], self.robot_handles[0])
        # Gets properties for all Dofs on an actor.
        dof_properties = self.gym.get_actor_dof_properties(
            self.envs[0], self.robot_handles[0]
        )
        # Gets state for the actor’s degrees of freedom
        # dof_states = self.gym.get_actor_dof_states(self.envs[0], self.robot_handles[0], gymapi.STATE_ALL)
        # Gets target velocity for the actor’s degrees of freedom
        # dof_velocity_targets = self.gym.get_actor_dof_velocity_targets(self.envs[0], self.robot_handles[0])

        return {
            "dof_count": dof_count,
            "dof_dict": dof_dict,
            "dof_names": dof_names,
            "dof_properties": dof_properties,
        }

    def create_env(self):
        # Load urdf asset
        self.robot_asset_root = self.asset_root
        self.robot_asset_file = self.config["envs"]["assets"]["assetFile"]
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = self.config["envs"]["assets"]["fix_base_link"]
        asset_options.disable_gravity = self.config["envs"]["assets"]["disable_gravity"]
        asset_options.flip_visual_attachments = self.config["envs"]["assets"][
            "flip_visual_attachments"
        ]
        asset_options.armature = self.config["envs"]["assets"]["armature"]
        asset_options.slices_per_cylinder = self.config["envs"]["assets"][
            "slices_per_cylinder"
        ]

        init_pose = self.config["envs"]["assets"]["init_pose"]
        self.robot_name = self.config["envs"]["assets"]["robot_name"]

        print_info(
            "Loading urdf asset {} from {}".format(
                self.robot_asset_file, self.robot_asset_root
            )
        )
        self.robot_asset = self.gym.load_asset(
            self.sim, self.robot_asset_root, self.robot_asset_file, asset_options
        )

        # Set up the env grid
        spacing = self.config["envs"]["envSpacing"]
        env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        envs = []
        robot_handles = []

        # configure env grid
        print("Creating %d environments" % self.num_envs)
        num_per_row = (
            int(self.num_envs_per_row)
            if self.num_envs_per_row is not None
            else int(math.sqrt(self.num_envs))
        )
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*init_pose[:3])
        pose.r = gymapi.Quat(*init_pose[3:7])
        for i in range(self.num_envs):
            # create env
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            envs.append(env)

            # add urdf
            # Create actor
            #     param1 (Env) – Environment Handle.
            #     param2 (Asset) – Asset Handle
            #     param3 (isaacgym.gymapi.Transform) – transform transform of where the actor will be initially placed
            #     param4 (str) – name of the actor
            #     param5 (int) – collision group that actor will be part of. The actor will not collide with anything
            #                    outside of the same collisionGroup
            #     param6 (int) – bitwise filter for elements in the same collisionGroup to mask off collision
            #     param7 (int) – segmentation ID used in segmentation camera sensors
            robot_handle = self.gym.create_actor(
                env,
                self.robot_asset,
                pose,
                self.robot_name,
                i,
                int(self.collision_mode),
                0,
            )
            self.gym.enable_actor_dof_force_sensors(env, robot_handle)
            robot_handles.append(robot_handle)

        self.envs = envs
        self.robot_handles = robot_handles

        # self._set_color()

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

    def _setup_attractors(self, traj, attr_rbs, attr_types, verbose=True):
        """
        Setup the attractors for tracking the trajectory using the embedded Isaac Gym PID controller

        :param traj: the trajectory to be tracked
        :param attr_rbs: link names to be attracted, same dim as traj
        :param attr_types: the type of the attractors
        :param verbose: if True, visualize the attractor spheres
        :return:
        """
        assert isinstance(attr_rbs, list), "The attracted joints should be a list"
        assert len(attr_rbs) == len(
            traj
        ), "The first dimension of trajectory should equal to attr_rbs"
        assert len(attr_rbs) == len(
            attr_types
        ), "The first dimension of attr_types should equal to attr_rbs"

        attractor_handles, axes_geoms, sphere_geoms = [], [], []
        for i in range(len(attr_rbs)):
            attractor_handle, axes_geom, sphere_geom = self._init_attractor(
                attr_rbs[i], attr_type=attr_types[i], verbose=verbose
            )
            attractor_handles.append(attractor_handle)
            axes_geoms.append(axes_geom)
            sphere_geoms.append(sphere_geom)
        return attr_rbs, attractor_handles, axes_geoms, sphere_geoms

    def add_tracking_target_sphere_axes(self):
        """
        Visualize the tracking target as a sphere with axes
        """
        from isaacgym import gymapi, gymutil

        # Create helper geometry used for visualization
        # Create a wireframe axis
        self.axes_geom = gymutil.AxesGeometry(0.1)
        # Create a wireframe sphere
        sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
        sphere_pose = gymapi.Transform(r=sphere_rot)
        self.sphere_geom = gymutil.WireframeSphereGeometry(
            0.03, 12, 12, sphere_pose, color=(1, 0, 0)
        )

    def add_marker(self, marker_pose):
        from isaacgym import gymapi

        asset_file = "mjcf/location_marker.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.density = 1.0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._marker_asset = self.gym.load_asset(
            self.sim, self.asset_root, asset_file, asset_options
        )

        self.marker_handles = []
        for i in range(self.num_envs):
            marker_handle = self.gym.create_actor(
                self.envs[i], self._marker_asset, marker_pose, "marker", i, 2, 0
            )
            self.gym.set_rigid_body_color(
                self.envs[i],
                marker_handle,
                0,
                gymapi.MESH_VISUAL,
                gymapi.Vec3(0.8, 0.0, 0.0),
            )
            self.marker_handles.append(marker_handle)

    def add_body_attached_camera(
        self, camera_props=None, attached_body=None, local_transform=None
    ):
        from isaacgym import gymapi

        self.camera_handle = self.gym.create_camera_sensor(self.envs[0], camera_props)
        body_handle = self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.robot_handles[0], attached_body
        )
        self.gym.attach_camera_to_body(
            self.camera_handle,
            self.envs[0],
            body_handle,
            local_transform,
            gymapi.FOLLOW_TRANSFORM,
        )

    def monitor_rigid_body_states(self):
        from isaacgym import gymtorch

        self._rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(self._rb_states)
        self.default_rb_states = self.rb_states.clone()

    def monitor_actor_root_states(self):
        from isaacgym import gymtorch

        self._root_states = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(self._root_states)
        self.default_root_states = self.root_states.clone()

    def monitor_dof_states(self):
        from isaacgym import gymtorch

        self._dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_states = gymtorch.wrap_tensor(self._dof_states)

    def monitor_robot_jacobian(self, robot_name=None):
        if robot_name is None:
            robot_name = self.robot_name
        from isaacgym import gymtorch

        self._jacobian = self.gym.acquire_jacobian_tensor(self.sim, robot_name)
        self.jacobian = gymtorch.wrap_tensor(self._jacobian)

    def monitor_robot_mass_matrix(self, robot_name=None):
        if robot_name is None:
            robot_name = self.robot_name
        from isaacgym import gymtorch

        self._massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, robot_name)
        self.massmatrix = gymtorch.wrap_tensor(self._massmatrix)

    def show(self, visual_obs_flag=False):
        """
        Visualize the urdf in an interactive viewer
        :param visual_obs_flag: If True, show the visual observation
        """
        from isaacgym import gymapi

        beauty_print(
            "Show the {} simulator in the interactive mode".format(self.robot_name),
            type="module",
        )

        if visual_obs_flag:
            fig = plt.figure("Visual observation", figsize=(8, 8))

        while not self.gym.query_viewer_has_closed(self.viewer):
            # Step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            # Step rendering
            self.gym.step_graphics(self.sim)

            if visual_obs_flag:
                # digest image
                self.gym.render_all_camera_sensors(self.sim)
                self.gym.start_access_image_tensors(self.sim)

                cam_img = self.gym.get_camera_image(
                    self.sim, self.envs[0], self.camera_handle, gymapi.IMAGE_COLOR
                ).reshape(1280, 1280, 4)
                cam_img = Im.fromarray(cam_img)
                plt.imshow(cam_img)
                plt.axis("off")
                plt.pause(1e-9)
                fig.clf()

                self.gym.end_access_image_tensors(self.sim)

            self.gym.draw_viewer(self.viewer, self.sim, False)
            self.gym.sync_frame_time(self.sim)

        beauty_print("Done")
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def get_num_bodies(self, robot_asset):
        num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        beauty_print(
            "The number of bodies in the urdf asset is {}".format(num_bodies),
            type="info",
        )
        return num_bodies

    def get_actor_rigid_body_info(self, actor_handle):
        rigid_body_dict = self.gym.get_actor_rigid_body_dict(self.envs[0], actor_handle)
        return rigid_body_dict

    def get_dof_info(self):
        # Gets number of Degree of Freedom for an actor
        dof_count = self.gym.get_actor_dof_count(self.envs[0], self.robot_handles[0])
        # maps degree of freedom names to actor-relative indices
        dof_dict = self.gym.get_actor_dof_dict(self.envs[0], self.robot_handles[0])
        # Gets forces for the actor’s degrees of freedom
        # dof_forces = self.gym.get_actor_dof_forces(self.envs[0], self.robot_handles[0])
        # Gets Frames for Degrees of Freedom of actor
        # dof_frames = self.gym.get_actor_dof_frames(self.envs[0], self.robot_handles[0])
        # Gets names of all degrees of freedom on actor
        dof_names = self.gym.get_actor_dof_names(self.envs[0], self.robot_handles[0])
        # Gets target position for the actor’s degrees of freedom.
        # dof_position_targets = self.gym.get_actor_dof_position_targets(self.envs[0], self.robot_handles[0])
        # Gets properties for all Dofs on an actor.
        dof_properties = self.gym.get_actor_dof_properties(
            self.envs[0], self.robot_handles[0]
        )
        # Gets state for the actor’s degrees of freedom
        # dof_states = self.gym.get_actor_dof_states(self.envs[0], self.robot_handles[0], gymapi.STATE_ALL)
        # Gets target velocity for the actor’s degrees of freedom
        # dof_velocity_targets = self.gym.get_actor_dof_velocity_targets(self.envs[0], self.robot_handles[0])

        return {
            "dof_count": dof_count,
            "dof_dict": dof_dict,
            "dof_names": dof_names,
            "dof_properties": dof_properties,
        }

    def get_robot_state(self, mode):
        from isaacgym import gymtorch

        if mode == "dof_force":
            # One force value per each DOF
            robot_dof_force = np.array(
                gymtorch.wrap_tensor(self.gym.acquire_dof_force_tensor(self.sim))
            )
            beauty_print("DOF forces:\n {}".format(robot_dof_force), 2)
            return robot_dof_force
        elif mode == "dof_state":
            # Each DOF state contains position and velocity and force sensor value
            for i in range(len(self.envs)):
                # TODO: multi envs
                robot_dof_force = np.array(
                    self.gym.get_actor_dof_forces(self.envs[i], self.robot_handles[i])
                ).reshape((-1, 1))
            robot_dof_pose_vel = np.array(
                gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim))
            )
            robot_dof_state = np.hstack((robot_dof_pose_vel, robot_dof_force))
            beauty_print("DOF states:\n {}".format(robot_dof_state), 2)
            return robot_dof_state
        elif mode == "dof_pose_vel":
            # Each DOF state contains position and velocity
            robot_dof_pose_vel = np.array(
                gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim))
            )
            beauty_print("DOF poses and velocities:\n {}".format(robot_dof_pose_vel), 2)
            return robot_dof_pose_vel
        elif mode == "dof_pose":
            # Each DOF pose contains position
            robot_dof_pose_vel = np.array(
                gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim))
            )
            beauty_print("DOF poses:\n {}".format(robot_dof_pose_vel[:, 0]), 2)
            return robot_dof_pose_vel[:, 0]
        elif mode == "dof_vel":
            # Each DOF velocity contains velocity
            robot_dof_pose_vel = np.array(
                gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim))
            )
            beauty_print("DOF velocities:\n {}".format(robot_dof_pose_vel[:, 1]), 2)
            return robot_dof_pose_vel[:, 1]
        elif mode == "dof_force_np":
            for i in range(len(self.envs)):
                # TODO: multi envs
                robot_dof_force = self.gym.get_actor_dof_forces(
                    self.envs[i], self.robot_handles[i]
                )
                beauty_print("DOF force s:\n {}".format(robot_dof_force), 2)
            return robot_dof_force
        else:
            raise ValueError("The mode {} is not supported".format(mode))

    def update_robot(
        self,
        traj,
        attractor_handles,
        axes_geom,
        sphere_geom,
        index,
        verbose=True,
        index_list=None,
    ):
        from isaacgym import gymutil

        for i in range(self.num_envs):
            # Update attractor target from current franka state
            attractor_properties = self.gym.get_attractor_properties(
                self.envs[i], attractor_handles[i]
            )
            pose = attractor_properties.target
            # pose.p: (x, y, z), pose.r: (w, x, y, z)
            if index_list is not None:
                index = index_list[i]
            pose.p.x = traj[index, 0]
            pose.p.y = traj[index, 1]
            pose.p.z = traj[index, 2]
            pose.r.w = traj[index, 6]
            pose.r.x = traj[index, 3]
            pose.r.y = traj[index, 4]
            pose.r.z = traj[index, 5]
            self.gym.set_attractor_target(self.envs[i], attractor_handles[i], pose)

            if verbose:
                # Draw axes and sphere at attractor location
                gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.envs[i], pose)
                gymutil.draw_lines(
                    sphere_geom, self.gym, self.viewer, self.envs[i], pose
                )

    def update_object(self, object_handles, object_poses, state_type):
        """
        Update the object pose

        :param object_handles:
        :param object_poses:
        :param state_type: gymapi.STATE_ALL, gymapi.STATE_POS, gymapi.STATE_VEL
        :return:
        """
        from isaacgym import gymapi

        for i in range(len(self.envs)):
            state = self.gym.get_actor_rigid_body_states(
                self.envs[i], object_handles[i], gymapi.STATE_NONE
            )
            state["pose"]["p"].fill(
                (object_poses[i][0], object_poses[i][1], object_poses[i][2])
            )
            state["pose"]["r"].fill(
                (
                    object_poses[i][3],
                    object_poses[i][4],
                    object_poses[i][5],
                    object_poses[i][6],
                )
            )
            state["vel"]["linear"].fill((0, 0, 0))
            state["vel"]["angular"].fill((0, 0, 0))
            self.gym.set_actor_rigid_body_states(
                self.envs[i], object_handles[i], state, state_type
            )

    def run_traj_multi_rigid_bodies(
        self,
        traj: List,
        attr_rbs: List = None,
        attr_types=None,
        object_start_pose: List = None,
        object_end_pose: List = None,
        object_related_joints: List = None,
        root_state=None,
        update_freq=0.001,
        verbose=True,
        index_list=None,
        recursive_play=False,
    ):
        """
        Set multiple attractors to let the urdf run the trajectory with multiple rigid bodies.

        :param traj: a list of trajectories, each trajectory is a numpy array of shape (N, 7)
        :param attr_rbs: [list], e.g. ["panda_left_hand", "panda_right_hand"]
        :param attr_types: [list], e.g. [gymapi.AXIS_ALL, gymapi.AXIS_ROTATION, gymapi.AXIS_TRANSLATION]
        :param object_start_pose: the initial pose of the object
        :param object_end_pose: the final pose of the object
        :param object_related_joints: the related joints of the object
        :param root_state: the root state of the urdf
        :param update_freq: the frequency of updating the urdf pose
        :param verbose: if True, visualize the attractor spheres
        :param index_list:
        :param recursive_play:
        :return:
        """
        from isaacgym import gymtorch

        assert (
            isinstance(traj, list) and len(traj) > 0
        ), "The trajectory should be a list of numpy arrays"
        beauty_print("Execute multi rigid bodies trajectory")

        self.gym.prepare_sim(self.sim)
        self.monitor_rigid_body_states()
        self.monitor_actor_root_states()
        self.monitor_dof_states()

        if root_state is not None:
            root_state = root_state.repeat(self.num_envs, 1, 1).reshape(
                (-1, self.num_envs, 13)
            )

        self.robot_root_states = self.root_states.view(self.num_envs, 1, -1)[..., 0, :]

        # Create the attractor
        attr_rbs, attr_handles, axes_geoms, sphere_geoms = self._setup_attractors(
            traj, attr_rbs, attr_types, verbose=verbose
        )

        # Time to wait in seconds before moving urdf
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

                if root_state is not None:
                    if index_list is not None:
                        root_state_tmp = torch.vstack(
                            [root_state[idx, 0] for idx in index_list]
                        )

                        self.gym.set_actor_root_state_tensor(
                            self.sim,
                            gymtorch.unwrap_tensor(
                                root_state_tmp.reshape(self.root_states.shape)
                            ),
                        )
                    else:
                        self.gym.set_actor_root_state_tensor(
                            self.sim,
                            gymtorch.unwrap_tensor(
                                root_state[index].reshape(self.root_states.shape)
                            ),
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


if __name__ == "__main__":
    print("1")
    robot_sim = RobotSim("CURI")
