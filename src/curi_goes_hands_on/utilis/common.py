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


def to_list(values):
    """Convert a series of values in various structure to a plain python list.

    Args:
        values (PoseStamped|Pose|Quaternion|list|tuple|np.ndarray): A data structure containing values to be
                                                                    put into a list.

    Returns:
        (list): A list of plain python number types.

    Raises:
        NotImplementedError: If the input type is not supported.
    """
    if isinstance(values, PoseStamped):
        output = to_list(values.pose)
    elif isinstance(values, Pose):
        output = [
            values.position.x,
            values.position.y,
            values.position.z,
            values.orientation.x,
            values.orientation.y,
            values.orientation.z,
            values.orientation.w,
        ]
    elif isinstance(values, Quaternion):
        output = [values.x, values.y, values.z, values.w]
    elif isinstance(values, list) or isinstance(values, tuple):
        output = list(values)
    elif isinstance(values, np.ndarray):
        output = values.tolist()
    else:
        raise NotImplementedError(
            "Type {} cannot be converted to list".format(type(values))
        )
    return output


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


def init_socket(ip, port):
    """Initialize a socket connection.

    Args:
        ip (str): IP for the socket server.
        port (int): Port of the server that receives the connection.

    Returns:
        (socket.socket): The socket object.
    """
    socket_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print_info("Connecting to socket server {}:{} ...".format(ip, port))
    # If the connection cannot be made, raise OSError
    socket_client.connect((ip, port))
    print_info("Socket connected on {}:{}".format(ip, port))
    return socket_client


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


def extend_right(s, max_length, filler=" "):
    """Given a string with length l_0, fill 'spaces' to its right such that its length becomes max_length.
    If l_0 > max_length,  no space will be filled and its length will be unchanged.

    Args:
        s (str): A string.
        max_length (int): Length of the output string.
        filler (str): A string to be filled on the right side of s, default space.

    Returns:
        (str)
    """
    return "{message:{filler}<{width}}".format(
        message=s, filler=filler, width=max_length
    )


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


def get_circle_centers_and_radii(img):
    """Get the center position and radius of circles in a numpy image.

    Args:
        img (np.ndarray): The image.

    Returns:
        (np.ndarray, np.ndarray): The center positions (N, 2) in pixel coordinate system for N circles;
                                  The radii (N, ) in pixel coordinate system for N circles.
    """
    assert isinstance(img, np.ndarray), print_error(
        f"Image type is not ndarray but {type(img)}"
    )
    centers = []
    radii = []
    gray_img = cv.medianBlur(cv.cvtColor(img, cv.COLOR_BGR2GRAY), 5)
    rows, columns, _ = get_image_hwc(gray_img)
    circles = cv.HoughCircles(
        gray_img,
        cv.HOUGH_GRADIENT,
        1,
        rows / 8,
        param1=100,
        param2=30,
        minRadius=80,
        maxRadius=300,
    )
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            centers.append([circle[0], circle[1]])
            radii.append([circle[2]])
        return centers, radii
    return None, None


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


def define_mask_rectangle_in_image(
        image, rectangle_width, rectangle_height, center_point, angle
):
    """In a rectangle, after rotating counterclockwise degree θ from the center point, the points coordinates are:
    x′ = (x0 － xcenter) cosθ － (y0 － ycenter) sinθ ＋ xcenter;
    y′ = (x0 － xcenter) sinθ ＋ (y0 － ycenter) cosθ ＋ ycenter;

    In opencv coordinate, the functions are refactored as
    x′ = (x0 － xcenter) cos(pi/180*θ) － (y0 － ycenter) sin(pi/180*θ) ＋ xcenter;
    y0 = row -y0
    ycenter = row -ycenter
    y′ = (x0 － xcenter) sin(pi/180*θ) ＋ (y0 － ycenter) cos(pi/180*θ) ＋ ycenter;
    y' = row - y'

    Args:
        image (np.ndarray): The image.
        rectangle_width (int): The width of the targeted rectangle.
        rectangle_height (int): The height of the targeted rectangle.
        center_point (list): The center point of the targeted rectangle.
        angle (float): The Rotation angle

    Returns:
        mask_rectangle: Mask of the targeted rectangle.
    """
    image_height, image_width, _ = get_image_hwc(image)
    mask_rectangle = np.zeros((image_height, image_width), dtype="uint8")
    rectangle_pts = np.array(
        [
            [
                center_point[0] - rectangle_height / 2,
                center_point[1] - rectangle_width / 2,
            ],
            [
                center_point[0] + rectangle_height / 2,
                center_point[1] - rectangle_width / 2,
            ],
            [
                center_point[0] + rectangle_height / 2,
                center_point[1] + rectangle_width / 2,
            ],
            [
                center_point[0] - rectangle_height / 2,
                center_point[1] + rectangle_width / 2,
            ],
        ],
    )
    center_point[1] = image_height - center_point[1]
    for i, points in enumerate(rectangle_pts):
        points[1] = image_height - points[1]
        # Convert image coordinates to plane coordinates
        rotated_rectangle_points_x = (
                (points[0] - center_point[0]) * np.cos(np.pi / 180.0 * angle)
                - (points[1] - center_point[1]) * np.sin(np.pi / 180.0 * angle)
                + center_point[0]
        )
        rotated_rectangle_points_y = (
                (points[0] - center_point[0]) * np.sin(np.pi / 180.0 * angle)
                + (points[1] - center_point[1]) * np.cos(np.pi / 180.0 * angle)
                + center_point[1]
        )
        # Convert plane coordinates to image coordinates
        rotated_rectangle_points_y = image_height - rotated_rectangle_points_y
        rectangle_pts[i] = [
            int(rotated_rectangle_points_x),
            int(rotated_rectangle_points_y),
        ]
    rectangle_pts = np.array(rectangle_pts, dtype=np.int32)
    cv.fillPoly(mask_rectangle, pts=[rectangle_pts], color=(1, 1, 1))

    return mask_rectangle


def append_two_dim_dict(target_dict, parent_key, child_key, child_value):
    """Append the two dim val into a dictionary

    Args:
        target_dict (dict): Dict
        parent_key: The first dim key
        child_key: The second dim key
        child_value:

    Returns:
        dict: {key_parent:{key_child:val}}
    """
    if parent_key in target_dict:
        target_dict[parent_key].update({child_key: child_value})
    else:
        target_dict.update({parent_key: {child_key: child_value}})


def get_patch_mask_bbox(mask):
    """Get the patch mask and bounding box from the image mask.

    Args:
        mask (list[list[bool|int]]):mask for the whole image with values be 0 (not masked) or 1 (masked).
               The derived mask is a (image_height, image_width) list.

    Returns:
        patch_mask (list[list[bool|int]]):Mask for the patch image with values be 0 (not masked) or 1 (masked).

        patch_bbox (list[list[bool|int]]):BBox list for the patch image. The list format is
                                                [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
    """
    foreground_mask_list = np.where(np.array(mask) > 0)
    top_left_x = min(foreground_mask_list[1])
    top_left_y = min(foreground_mask_list[0])
    bottom_right_x = max(foreground_mask_list[1])
    bottom_right_y = max(foreground_mask_list[0])
    patch_bbox = [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
    patch_mask = mask[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    return patch_mask, patch_bbox


def find_two_masks_intersection_area(mask_a, mask_b):
    """Finds whether there is overlapping between the two masks.
    The return represents the number of pixels in the overlapped area.

    Args:
       mask_a (list[list[bool|int]]): Mask for the whole image with values be 0 (not masked) or 1 (masked).
       mask_b (list[list[bool|int]]): Mask for the whole image with values be 0 (not masked) or 1 (masked).

    Returns:
       int: The number of pixels at the intersection of the two masks.
    """
    return get_mask_area(np.bitwise_and(mask_a, mask_b))


def get_mask_area(mask):
    """Acquire the number of pixels in the masked area.

    Args:
        mask (list[list[bool|int]]): The mask whose values are either 0 (not masked) or 1 (masked) for the entire image.

    Returns:
        int: The number of masked pixels.
    """
    return len(np.argwhere(mask == 1)[:, ::-1])


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
