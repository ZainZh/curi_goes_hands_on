# -*- coding: utf-8 -*-
# @Auther   : Zheng SUN (ZainZh)
# @Time     : 2025/1/4
# @Address  : clover Lab @ CUHK
# @FileName : function_map.py

# @Description : TODO
import pyzed.sl as sl

function_map = {
    "HD2K": sl.RESOLUTION.HD2K,
    "HD1080": sl.RESOLUTION.HD1080,
    "HD720": sl.RESOLUTION.HD720,
    "VGA": sl.RESOLUTION.VGA,
    "Ultra": sl.DEPTH_MODE.ULTRA,
    "Quality": sl.DEPTH_MODE.QUALITY,
    "Performance": sl.DEPTH_MODE.PERFORMANCE,
}
