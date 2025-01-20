# -*- coding: utf-8 -*-
# @Auther   : Zheng SUN (ZainZh)
# @Time     : 2025/1/4
# @Address  : clover Lab @ CUHK
# @FileName : common.py

# @Description : common non-class functions for projects

import rospy
import socket
import time

import cv2 as cv
import numpy as np
import os.path as osp

from functools import wraps
from omegaconf import OmegaConf
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from sensor_msgs.msg import CompressedImage


def _preprocess_print(*args):
    """Preprocess the input for colorful printing.

    Args:
        args (Any|None): One or more any type arguments to print.

    Returns:
        str Msg to print.
    """
    str_args = ""
    for a in args:
        if isinstance(a, np.ndarray):
            str_args += "\n" + np.array2string(a, separator=", ")
        else:
            str_args += " " + str(a)
    separate_with_newline = str_args.split("\n")
    extra_whitespaces_removed = []
    for b in separate_with_newline:
        extra_whitespaces_removed.append(" ".join(b.split()))
    return "\n".join(extra_whitespaces_removed)


def print_debug(*args):
    """Print information with green."""
    print("".join(["\033[1m\033[92m", _preprocess_print(*args), "\033[0m"]))


def print_info(*args):
    """Print information with sky blue."""
    print("".join(["\033[1m\033[94m", _preprocess_print(*args), "\033[0m"]))


def print_warning(*args):
    """Print a warning with yellow."""
    print("".join(["\033[1m\033[93m", _preprocess_print(*args), "\033[0m"]))


def print_error(*args):
    """Print error with red."""
    print("".join(["\033[1m\033[91m", _preprocess_print(*args), "\033[0m"]))


def get_ros_param(name, value=None, verbose_level=2):
    """Get private/public ROS param from the param server.

    Args:
        name (str): Param name with prefix ~, / or no prefix. The '~' explicitly indicates that
                    the param is a private param for the node calling this function. In this case,
                    the full name of the param is prefixed with the node's name.
                    The `/` indicates a public param and its name is not prefixed by the node's name.
                    When there is no prefix, it will first be searched in private params, and then in
                    public params. If with a leading ~, it will only be searched in private params.
        value (Any|None): Any Return value if param is not set.
        verbose_level (int): 0 and below zero for debug, 1 for info, 2 and above for warning.

    Returns:
        (Any): Param value.
    """
    name = name.lstrip("/")
    private = "~%s" % name
    if rospy.has_param(private):
        if verbose_level <= 0:
            print_debug(f"Private param '{name}' set as {rospy.get_param(private)}")
        return rospy.get_param(private)
    elif rospy.has_param(name):
        if verbose_level <= 0:
            print_debug(f"Public param '{name}' set as {rospy.get_param(name)}")
        return rospy.get_param(name)
    else:
        if verbose_level <= 1:
            print_info(f"'{name}' not exist, the default '{value}' is returned")
        return value


def omega_to_list(omega):
    """Put a set of values given in a OmegaConfig into a list.

    Args:
        omega:

    Returns:

    """
    return OmegaConf.to_object(omega)


def load_omega_config(config_name):
    """Load the configs listed in config_name.yaml.

    Args:
        config_name (str): Name of the config file.

    Returns:
        (dict): A dict of configs.
    """
    return OmegaConf.load(
        osp.join(osp.dirname(__file__), "../../config/{}.yaml".format(config_name))
    )


def update_omega_config(config_name, key, value):
    """Update the content of the config file config_name.yaml in the predefined path.
    If the configuration file has not yet been created, create the file and write the specified key value into it.

    If it has been created:
    1) If the `key` is new to the existed configurations, create this key and assign its value with `value`.
    2) If the `key` is already there, override its value with `value`.

    Other keys and their values are not affected, but are passed on to the newly created configuration file which
    overwrites the original file.

    Args:
        config_name (str): Name of the config file not including suffix (.yaml).
        key (Any): Key of the item.
        value (Any): Value of the item whose key is `key`.
    """
    if isinstance(value, np.ndarray):
        value = value.tolist()
    config_item = OmegaConf.create({key: value})
    config_file = osp.join(osp.dirname(__file__), f"../../config/{config_name}.yaml")
    if osp.exists(config_file):
        loaded = OmegaConf.load(config_file)
    else:
        loaded = None
    if loaded:
        if key in loaded:
            OmegaConf.update(loaded, key, value)
        else:
            loaded = OmegaConf.merge(config_item, loaded)
    else:
        loaded = config_item
    OmegaConf.save(loaded, f=config_file)


def sd_position(position):
    """Standardize a position-like object to a (3,) ndarray whose elements represent x, y, and z.

    Args:
        position (np.ndarray|list|Point): A position-like data object.

    Returns:
        np.ndarray: (3,) [x, y, z].

    Raises:
        NotImplementedError: If the input type or shape is not supported.
    """
    if isinstance(position, np.ndarray):
        if position.shape == (3,):
            return position
        elif position.shape == (4, 4):
            return position[:3, 3]
        else:
            raise NotImplementedError
    elif isinstance(position, list) or isinstance(position, tuple):
        return sd_position(np.array(position))
    elif isinstance(position, Point):
        return sd_position(np.array([position.x, position.y, position.z]))
    else:
        raise NotImplementedError


def pretty_print_configs(cfgs):
    """Print a dict of configurations in a visual friendly and organized way.

    Args:
        cfgs (dict): A dict of configures. The items could be string, number, or a list/tuple.

    Returns:
        None
    """
    max_key_len = 0
    max_value_len = 0
    for key, value in cfgs.items():
        key_str = "{}".format(key)
        if len(key_str) > max_key_len:
            max_key_len = len(key_str)
        if isinstance(value, list) or isinstance(value, tuple):
            for i in value:
                i_str = "{}".format(i)
                if len(i_str) > max_value_len:
                    max_value_len = len(i_str)
        else:
            value_str = "{}".format(value)
            if len(value_str) > max_value_len:
                max_value_len = len(value_str)

    print_info(
        "\n{}{}{}".format(
            "=" * (max_key_len + 1), " ROPORT CONFIGS ", "=" * (max_value_len - 15)
        )
    )
    for key, value in cfgs.items():
        key_msg = "{message: <{width}}".format(message=key, width=max_key_len)
        empty_key_msg = "{message: <{width}}".format(message="", width=max_key_len)
        if isinstance(value, list) or isinstance(value, tuple):
            for i, i_v in enumerate(value):
                if i == 0:
                    print_info("{}: {}".format(key_msg, i_v))
                else:
                    print_info("{}: {}".format(empty_key_msg, i_v))
        else:
            print_info("{}: {}".format(key_msg, value))
    print_info(
        "{}{}{}\n".format(
            "=" * (max_key_len + 1), " END OF CONFIGS ", "=" * (max_value_len - 15)
        )
    )


def is_array_like(array):
    """"""
    if isinstance(array, str):
        return False
    return hasattr(array, "__len__") and hasattr(array, "__iter__")


def expect_any_input(hint):
    """Get any kind of user input.

    Args:
        hint (str): Hint to the user.

    Returns:
        Any user input.
    """
    if not isinstance(hint, str):
        hint = str(hint)
    if not hint.endswith(" "):
        hint += " "
    return input(hint)


def expect_yes_no_input(hint, is_yes_default=True):
    """Get user input for a yes/no choice.

    Args:
        hint: str Hint for the user to input.
        is_yes_default: bool If true, 'yes' will be considered as the default when empty input was given.
                        Otherwise, 'no' will be considered as the default choice.

    Returns:
        bool If the choice matches the default.
    """
    if is_yes_default:
        suffix = "(Y/n):"
        default = "yes"
    else:
        suffix = "(y/N):"
        default = "no"
    flag = input(" ".join((hint, suffix))).lower()

    expected_flags = ["", "y", "n"]
    while flag not in expected_flags:
        print_warning(
            f"Illegal input {flag}, valid inputs should be Y/y/N/n or ENTER for the default: {default}"
        )
        return expect_yes_no_input(hint, is_yes_default)

    if is_yes_default:
        return flag != "n"
    else:
        return flag != "y"


def is_float_compatible(string):
    """Check if the string can be converted to a float.

    Args:
        string (str): Input string.

    Returns:
        bool: True if the string can be converted to a float, false otherwise.
    """
    string = string.lstrip("-")
    s_dot = string.split(".")
    if len(s_dot) > 2:
        return False
    s_e_plus = string.split("e+")
    if len(s_e_plus) == 2:
        return is_float_compatible("".join(s_e_plus))
    s_e_minus = string.split("e-")
    if len(s_e_minus) == 2:
        return is_float_compatible("".join(s_e_minus))
    s_e = string.split("e")
    if len(s_e) == 2:
        return is_float_compatible("".join(s_e))

    for si in s_dot:
        if not si.isdigit():
            return False
    return True


def expect_float_input(hint):
    """Get user input for obtaining a float number.

    Args:
        hint (str): Hint for the user to input.

    Returns:
        float: User input value.
    """
    user_input = expect_any_input(hint)
    while not is_float_compatible(user_input):
        print_warning(
            f"Illegal input '{user_input}', valid input should be a float number"
        )
        return expect_float_input(hint)
    return float(user_input)


def to_compressed_image(array):
    """Convert a np array to a CompressedImage msg.

    Args:
        array (np.ndarray): A Numpy array representing an image.

    Returns:
        CompressedImage: Converted sensor_msgs/CompressedImage msg.
    """
    image_msg = CompressedImage()
    image_msg.header.stamp = rospy.Time.now()
    image_msg.format = "jpeg"
    image_msg.data = np.array(cv.imencode(".jpg", array)[1]).tobytes()
    return image_msg


def get_image_hwc(image):
    """Get the height, width, and channel of a ndarray image.

    Args:
        image (np.ndarray): The image.

    Returns:
        (int, int, int): Height, width, and channel number of the image.
    """
    assert isinstance(image, np.ndarray), print_error(
        f"Image type is not ndarray but {type(image)}"
    )
    if len(image.shape) == 2:
        h, w = image.shape
        c = 1
    elif len(image.shape) == 3:
        if image.shape[0] == 3:
            h, w = image.shape[1:]
        elif image.shape[-1] == 3:
            h, w = image.shape[:2]
        else:
            raise ValueError(f"Image of shape {image.shape} is not supported")
        c = 3
    else:
        raise ValueError(f"Image of shape {image.shape} is not supported")
    return h, w, c


def timeit(func):
    """A timer decorator."""

    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        frequency = 1.0 / total_time
        print_info(
            f"Function {func.__name__} took {total_time:.2f} seconds ({frequency:.2f} Hz)"
        )
        return result

    return timeit_wrapper


def download_file(remote_url, local_path):
    """Download a file to local path from the remote url.

    Args:
        remote_url (str):
        local_path (str):

    Returns:
        None
    """
    from urllib import request

    request.urlretrieve(remote_url, local_path)


def list_to_pose(pose_list):
    """Convert a list to a Pose msg.

    Args:
        pose_list (list): A list of 7 elements representing a pose.

    Returns:
        Pose: A Pose msg.
    """
    assert len(pose_list) == 7, print_error(
        f"Length of the input list {len(pose_list)} is not 7"
    )
    pose = Pose()
    pose.position = Point(*pose_list[:3])
    pose.orientation = Quaternion(*pose_list[3:])
    return pose
