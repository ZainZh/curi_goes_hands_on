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

class MoveItCURI(object):
    def __init__(self, config="moveit_curi"):
        super(MoveItCURI, self).__init__()
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('moveit_curi', anonymous=True)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group_name = "panda_arm"
        self.group = moveit_commander.MoveGroupCommander(self.group_name)
        self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                            moveit_msgs.msg.DisplayTrajectory,
                                                            queue_size=20)
        self.planning_frame = self.group.get_planning_frame()
        self.eef_link = self.group.get_end_effector_link()
        self.group_names = self.robot.get_group_names()
        self.box_name = ''
        self.box_pose = geometry_msgs.msg.PoseStamped()
        self.box_pose.header.frame_id = "panda_link0"
        self.box_pose.pose.position.x = 0.4
        self.box_pose.pose.position.y = 0.0
        self.box_pose.pose.position.z = 0.5
        self.box_name = "box"
        self.scene.add_box(self.box_name, self.box_pose, size=(0.1, 0.1, 0.1))
        self.group.set_max_velocity_scaling_factor(0.1)
        self.group.set_max_acceleration_scaling_factor(0.1)
        self.group.set_goal_position_tolerance(0.01)
        self.group.set_goal_orientation_tolerance(0.01)
        self.group.set_planning_time(5)
        self.group.set_num_planning_attempts(5)
        self.group.set_planner_id("RRTConnectkConfigDefault")
        self.group.set_planning_time(5)
        self.group.set_num_planning_attempts(5)
        self.group.set_planner_id("RRTConnectkConfigDefault")
        self.group.set_planning_time(5)
        self.group.set_num_planning_attempts(5)
        self.group.set_planner_id("RRTConnectkConfigDefault")
        self.group.set_planning_time(5)
        self.group.set_num_planning_attempts(5)
        self.group.set_planner_id("RRTConnectkConfigDefault")
        self.group.set_planning_time(5)
        self.group.set_num_planning_attempts(5)