# -*- coding: utf-8 -*-
# @Auther   : Zheng SUN (ZainZh)
# @Time     : 2025/1/4
# @Address  : clover Lab @ CUHK
# @FileName : zed.py

# @Description : TODO
import pyzed.sl as sl
from utilis.common import load_omega_config, print_info
from function_map import function_map
import os


class ZED(object):
    def __init__(self, config="zed"):
        super().__init__()
        self.zed = sl.Camera()
        # Load the configuration file
        self.config = load_omega_config(config)

        # Set configuration parameters
        self.init_params = sl.InitParameters()
        self.runtime_params = sl.RuntimeParameters()
        self.image_save_path = None
        self.load_params()

    def load_params(self):
        resolution = function_map[self.config["resolution"]]
        fps = self.config["fps"]
        depth_mode = function_map[self.config["depth_mode"]]

        self.image_save_path = self.config["image_save_path"]
        if not os.path.exists(self.image_save_path):
            os.makedirs(self.image_save_path)
        self.init_params.camera_resolution = resolution
        self.init_params.camera_fps = fps
        self.init_params.depth_mode = depth_mode

    def __del__(self):
        self.close_camera()

    def open_camera(self):
        err = self.zed.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            exit()

    def capture_image(self):
        """
        Capture the image
        :return:
        image: sl.Mat
        """
        image = sl.Mat()
        if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(image, sl.VIEW.LEFT)
            # timestamp = self.zed.get_timestamp(sl.TIME_REFERENCE.IMAGE)
            # print_info(
            #     "Image resolution: {0} x {1} || Image timestamp: {2}".format(image.get_width(), image.get_height(),
            #                                                                  timestamp.get_milliseconds()))
            return image

    def capture_depth_image(self, is_save=False):
        """
        Capture the depth image
        :return:
        image: sl.Mat
        depth_map: sl.Mat
        point_cloud: sl.Mat
        """

        image = sl.Mat()
        depth_map = sl.Mat()
        point_cloud = sl.Mat()
        if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(image, sl.VIEW.LEFT)
            self.zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)
            self.zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            if is_save:
                image.write(f"{self.image_save_path}/Image.png")
                depth_map.write(f"{self.image_save_path}/Depth.png")
                point_cloud.write(f"{self.image_save_path}/Pointcloud.ply")
                print_info("Image files saving succeed")

            return image, depth_map, point_cloud

    @staticmethod
    def retrieve_depth_value(pixel, depth_map):
        """
        Retrieve the depth value of the point
        :param pixel: [int, int]
        :param depth_map: sl.Mat
        :return:
        depth_value: float
        """
        depth_value = depth_map.get_value(pixel[0], pixel[1])
        return depth_value

    def close_camera(self):
        self.zed.close()

    def run(self):
        pass


if __name__ == "__main__":
    zed = ZED()
    zed.open_camera()
    image, depth_map, point_cloud = zed.capture_depth_image(is_save=True)
    zed.close_camera()
