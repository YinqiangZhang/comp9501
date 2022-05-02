#! /usr/bin/env python

__author__ = "Niels Mündler"
__copyright__ = ""
__credits__ = [""]
__version__ = "1.0"
__maintainer__ = "Xiao Wang"
__email__ = "xiao.wang@tum.de"
__status__ = "Released"

"""
Extracts UtmOrigins and Speed limits for intersections
"""

from pathlib import Path

from commonroad_rl.utils_ind.utils.tracks_import import read_meta_info

base_path = "../data/"

already_found_ids = set()
for file_name in Path(base_path).glob("*_recordingMeta.csv"):
    meta_info = read_meta_info(file_name)
    if meta_info["locationId"] not in already_found_ids:
        print(
            "{}: UtmOrigin [{}, {}]".format(
                meta_info["locationId"],
                meta_info["xUtmOrigin"],
                meta_info["yUtmOrigin"],
            )
        )
        print(
            "{}: Speed limit {}".format(
                meta_info["locationId"], meta_info["speedLimit"]
            )
        )
        already_found_ids.add(meta_info["locationId"])
