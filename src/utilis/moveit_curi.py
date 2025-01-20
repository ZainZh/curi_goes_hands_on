# -*- coding: utf-8 -*-
# @Auther   : Zheng SUN (ZainZh)
# @Time     : 2025/1/16
# @Address  : clover Lab @ CUHK
# @FileName : moveit_curi.py

# @Description : Trajectory planning with MoveIt! for CURI robot, Dual-arm robot with 7 DOF each arm.
from __future__ import print_function
from six.moves import input
import rospy
import sys
import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg as moveit_msgs
import geometry_msgs.msg as geometry_msgs
from math import pi, tau, dist, fabs, cos
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
from shape_msgs.msg import SolidPrimitive
from common import load_omega_config, print_info, print_debug, print_warning
from typing import List
import moveit_ros_planning_interface


class MoveItCURI(object):
    def __init__(self, config="moveit_curi"):
        super(MoveItCURI, self).__init__()
        self.config = load_omega_config(config)
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("moveit_curi", anonymous=True)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.scene_pub = rospy.Publisher(
            "planning_scene", moveit_msgs.PlanningScene, queue_size=10
        )

        self.left_group_name = self.config["left_arm_group"]
        self.left_group = moveit_commander.MoveGroupCommander(self.left_group_name)
        self.left_eef_link = self.left_group.get_end_effector_link()
        self.group_names = self.robot.get_group_names()

        self.right_group_name = self.config["right_arm_group"]
        self.right_group = moveit_commander.MoveGroupCommander(self.right_group_name)
        self.right_eef_link = self.right_group.get_end_effector_link()
        # self.right_group.set_named_target("right_arm_home")

        self.dual_group_name = self.config["dual_arm_group"]
        self.dual_group = moveit_commander.MoveGroupCommander(self.dual_group_name)
        self.default_dual_arm_joint_state = self.config["default_dual_arm_joint_state"]

        self.left_hand_group_name = self.config["left_hand_group"]
        self.right_hand_group_name = self.config["right_hand_group"]
        self.left_hand = moveit_commander.MoveGroupCommander(self.left_hand_group_name)
        self.right_hand = moveit_commander.MoveGroupCommander(
            self.right_hand_group_name
        )

        self.init_state()

    def __del__(self):
        moveit_commander.roscpp_shutdown()
        moveit_commander.os._exit(0)

    def go_default_pose(self):
        self.left_group.set_named_target("default_pose")
        self.right_group.set_named_target("default_pose")
        self.left_hand.set_named_target("close")
        self.right_hand.set_named_target("close")
        self.left_group.go()
        self.right_group.go()
        self.left_hand.go()
        self.right_hand.go()

    def init_state(self):
        # self.init_scene()
        # self.left_group.set_goal_position_tolerance(0.01)
        # self.left_group.set_goal_orientation_tolerance(0.01)
        # self.left_group.set_planning_time(5)

        self.go_default_pose()
        self.dual_group.allow_replanning(True)

    def init_scene(self):
        """
        Clear the scene and add the table
        """
        table_object = moveit_msgs.CollisionObject()
        table_object.id = "table"
        table_object.header.frame_id = "summit_xls_base_footprint"

        # 定义盒子的几何形状
        table_primitive = SolidPrimitive()
        table_primitive.type = SolidPrimitive.BOX
        table_primitive.dimensions = [4, 2, 1]

        table_pose = geometry_msgs.Pose()
        table_pose.position.x = 5
        table_pose.position.y = 5
        table_pose.position.z = 0.5

        table_object.primitives.append(table_primitive)
        table_object.primitive_poses.append(table_pose)
        table_object.operation = table_object.ADD
        planning_scene = moveit_msgs.PlanningScene()
        planning_scene.is_diff = True
        planning_scene.world.collision_objects.append(table_object)

        planning_scene_interface = (
            moveit_ros_planning_interface.PlanningSceneInterface()
        )
        planning_scene_interface.applyCollisionObject(table_object)

    def go_to_joint_state(self, joint_state, group_name="left_arm"):
        """
        Go to the joint state
        Args:
            joint_state: list[float] (7,)
            group_name: str
        """
        if group_name == "left_arm":
            group = self.left_group
        else:
            group = self.right_group
        group.go(joint_state, wait=True)
        group.stop()

    @property
    def left_arm_joint_state(self) -> List[float]:
        """
        Get the current joint values of the left arm
        Returns:
            list[float] (7,)
        """

        return self.left_group.get_current_joint_values()

    @property
    def right_arm_joint_state(self) -> List[float]:
        """
        Get the current joint values of the right arm
        Returns:
            list[float] (7,)
        """
        return self.right_group.get_current_joint_values()

    @property
    def left_arm_pose(self) -> List[float]:
        """
        Get the current pose of the left arm
        Returns
            List[float] (x, y, z, qx, qy, qz, qw)
        -------

        """
        return pose_to_list(self.left_group.get_current_pose().pose)

    @property
    def right_arm_pose(self) -> List[float]:
        """
        Get the current pose of the right arm
        Returns
            List[float] (x, y, z, qx, qy, qz, qw)
        -------

        """
        return pose_to_list(self.right_group.get_current_pose().pose)

    @property
    def dual_arm_joint_state(self) -> List[float]:
        """
        
        Returns
        -------

        """ """

        """
        return self.dual_group.get_current_joint_values()

    @property
    def dual_arm_pose(self) -> List[float]:
        """
        
        Returns
        -------

        """ """

        """
        return pose_to_list(self.dual_group.get_current_pose().pose)

    def is_all_close(self, goal, actual, tolerance):
        """
        Convenience method for testing if the values in two lists are within a tolerance of each other.
        For Pose and PoseStamped inputs, the angle between the two quaternions is compared (the angle
        between the identical orientations q and -q is calculated correctly).
        @param: goal       A list of floats, a Pose or a PoseStamped
        @param: actual     A list of floats, a Pose or a PoseStamped
        @param: tolerance  A float
        @returns: bool
        """
        if type(goal) is list:
            for index in range(len(goal)):
                if abs(actual[index] - goal[index]) > tolerance:
                    return False

        elif type(goal) is geometry_msgs.PoseStamped:
            return self.is_all_close(goal.pose, actual.pose, tolerance)

        elif type(goal) is geometry_msgs.Pose:
            x0, y0, z0, qx0, qy0, qz0, qw0 = pose_to_list(actual)
            x1, y1, z1, qx1, qy1, qz1, qw1 = pose_to_list(goal)
            # Euclidean distance
            d = dist((x1, y1, z1), (x0, y0, z0))
            # phi = angle between orientations
            cos_phi_half = fabs(qx0 * qx1 + qy0 * qy1 + qz0 * qz1 + qw0 * qw1)
            return d <= tolerance and cos_phi_half >= cos(tolerance / 2.0)

        return True

    def left_arm_go_to_joint_state(self):
        """
        Go to the joint state of the left arm
        """
        joint_goal = self.left_group.get_current_joint_values()
        joint_goal[0] = 0
        joint_goal[1] = -tau / 8
        joint_goal[2] = 0
        joint_goal[3] = -tau / 4
        joint_goal[4] = 0
        joint_goal[5] = tau / 6  # 1/6 of a turn
        joint_goal[6] = 0
        self.left_group.go(joint_goal, wait=True)
        self.left_group.stop()

        current_joints = self.left_group.get_current_joint_values()
        return self.is_all_close(joint_goal, current_joints, 0.01)

    def right_arm_go_to_joint_state(self):
        """
        Go to the joint state of the right arm
        """
        joint_goal = self.right_group.get_current_joint_values()
        joint_goal[0] = 0
        joint_goal[1] = -tau / 8
        joint_goal[2] = 0
        joint_goal[3] = -tau / 4
        joint_goal[4] = 0
        joint_goal[5] = tau / 6  # 1/6 of a turn
        joint_goal[6] = 0
        self.right_group.go(joint_goal, wait=True)
        self.right_group.stop()

        current_joints = self.right_group.get_current_joint_values()
        return self.is_all_close(joint_goal, current_joints, 0.01)

    def dual_arm_go_to_joint_state_test(self):
        """
        Go to the joint state of the dual arm
        input: joint_state: list[float] (14,)

        Output: bool
        """
        joint_goal_left = self.left_group.get_current_pose()
        joint_goal_right = self.right_group.get_current_pose()

        joint_goal_left.pose.position.y += 0.1
        joint_goal_right.pose.position.y += 0.1

        self.dual_group.set_pose_target(joint_goal_left, self.left_eef_link)
        self.dual_group.set_pose_target(joint_goal_right, self.right_eef_link)
        traj = self.dual_group.plan()
        self.dual_group.execute(traj[1])
        rospy.sleep(1)
        self.go_default_pose()


if __name__ == "__main__":
    moveit_curi = MoveItCURI()
    print("left_arm_joint_state: ", moveit_curi.left_arm_joint_state)
    print("right_arm_joint_state: ", moveit_curi.right_arm_joint_state)
    print("dual_arm_joint_state: ", moveit_curi.dual_arm_joint_state)
    print("left_arm_pose: ", moveit_curi.left_arm_pose)
    print("right_arm_pose: ", moveit_curi.right_arm_pose)
    # print("dual_arm_pose: ", moveit_curi.dual_arm_pose)

    moveit_curi.dual_arm_go_to_joint_state_test()
    # print(moveit_curi.left_arm_go_to_joint_state())
    # print(moveit_curi.right_arm_go_to_joint_state())
    # print(moveit_curi.right_arm_go_to_joint_state())
    # print(moveit_curi.dual_arm_go_to_joint_state())
