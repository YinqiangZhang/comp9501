#! /usr/bin/env python

__author__ = "Niels Mündler"
__copyright__ = ""
__credits__ = [""]
__version__ = "1.0"
__maintainer__ = "Xiao Wang"
__email__ = "xiao.wang@tum.de"
__status__ = "Released"

__desc__ = """
Translate converted inD intersection lanelet to useful coordinates, translating them into local coordinate system (also better usable by CommonRoad)
"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from commonroad.common.file_writer import (
    CommonRoadFileWriter,
    OverwriteExistingFile,
    Tag,
)
from commonroad.common.file_reader import CommonRoadFileReader

DOWNLOADED = "downloaded"
ORIGINAL = "original"

locationId_to_name = {
    1: "Bendplatz",
    2: "frankenberg",
    3: "heckstrasse",
    4: "aseag",
}

# as extracted from recordingMeta files
locationId_to_UTMorigin = {
    1: np.array([293487.1224, 5629711.58163]),
    2: np.array([295620.9575, 5628102.04258]),
    3: np.array([300127.0853, 5629091.0587]),
    4: np.array([297631.31870, 5629917.34465]),
}

# speed limit [m/s] = 50kmh
# as extracted from recordingMeta files (same in all recordings)
speed_limit = 13.88889


def get_parser():
    parser = ArgumentParser(description=__desc__)
    parser.add_argument(
        "-i",
        default="../inD_LaneletMaps/convert_orig",
        help="Path to directory containing commonroad formatted lanelets (converts all files)",
        dest="input",
    )
    parser.add_argument(
        "-o", default=".", help="Path to output directory", dest="output"
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()

    input_files = []

    input_d = Path(args.input)
    if not input_d.exists():
        print("Error: path {} does not exist".format(input_d))
        exit(-1)
    if not input_d.is_dir():
        print("Error: path {} is not a directory".format(input_d))
        exit(-1)

    output_d = Path(args.output)
    output_d.mkdir(parents=True, exist_ok=True)

    # TODO read these values from the original scenario automatically
    author = (
        "Julian Bock, Robert Krajewski, Lennart Vater, Lutz Eckstein, Niels Mündler"
    )
    affiliation = "RWTH Aachen University, Germany"
    source = "https://www.ind-dataset.com/"
    tags = {Tag.INTERSECTION, Tag.MULTI_LANE, Tag.SPEED_LIMIT, Tag.URBAN}

    translation_dict = locationId_to_UTMorigin

    for i, location_name in locationId_to_name.items():
        # read
        scenario, pps = CommonRoadFileReader(
            input_d.joinpath("{}.xml".format(location_name))
        ).open()

        # translate
        scenario.translate_rotate(-translation_dict[i], 0)
        # set speed limit
        for lanelet in scenario.lanelet_network.lanelets:
            lanelet._speed_limit = (
                speed_limit  # dirty hack to mutate immutable speed limit
            )

        # write
        fw = CommonRoadFileWriter(scenario, pps, author, affiliation, source, tags)
        fw.write_to_file(
            str(output_d.joinpath("{}.xml".format(location_name))),
            OverwriteExistingFile.ALWAYS,
        )
