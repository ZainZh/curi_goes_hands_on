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
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi, tau, dist, fabs, cos
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
from common import load_omega_config


class MoveItCURI(object):
    def __init__(self, config="moveit_curi"):
        super(MoveItCURI, self).__init__()
        self.config = load_omega_config(config)
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("moveit_curi", anonymous=True)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()

        self.left_group_name = "left_arm"
        self.left_group = moveit_commander.MoveGroupCommander(self.left_group_name)
        self.left_eef_link = self.left_group.get_end_effector_link()
        self.group_names = self.robot.get_group_names()

        self.right_group_name = "right_arm"
        self.right_group = moveit_commander.MoveGroupCommander(self.right_group_name)
        self.right_eef_link = self.right_group.get_end_effector_link()

    @property
    def left_arm_joint_state(self):
        return self.left_group.get_current_joint_values()

    @property
    def right_arm_joint_state(self):
        return self.right_group.get_current_joint_values()

    @property
    def left_arm_pose(self):
        return pose_to_list(self.left_group.get_current_pose().pose)

    @property
    def right_arm_pose(self):
        return pose_to_list(self.right_group.get_current_pose().pose)


if __name__ == "__main__":
    moveit_curi = MoveItCURI()
    print("left_arm_joint_state: ", moveit_curi.left_arm_joint_state)
    print("right_arm_joint_state: ", moveit_curi.right_arm_joint_state)
    print("left_arm_pose: ", moveit_curi.left_arm_pose)
    print("right_arm_pose: ", moveit_curi.right_arm_pose)
