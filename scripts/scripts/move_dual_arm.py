# -*- coding: utf-8 -*-
# @Auther   : Zheng SUN (ZainZh)
# @Time     : 2025/1/17
# @Address  : clover Lab @ CUHK
# @FileName : move_dual_arm.py

# @Description : TODO
from moveit_python.move_group_interface import MoveGroupInterface
import rospy

if __name__ == "__main__":
    move_group_interfece = MoveGroupInterface("arm_with_torso", "base_link")
