import os
from glob import glob

from setuptools import find_packages, setup

package_name = "vision_pkg_ros2"


def package_files(directory):
    files = []
    for path in glob(os.path.join(directory, "**", "*"), recursive=True):
        if os.path.isfile(path):
            install_dir = os.path.join("share", package_name, os.path.dirname(path))
            files.append((install_dir, [path]))
    return files


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/vision.launch.py"]),
    ] + package_files("models") + package_files("docs"),
    install_requires=["setuptools"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="xytron",
    maintainer_email="xytron@todo.todo",
    description="ROS 2 vision package for lane, stop-line, and traffic-light detection.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "lane_node = vision_pkg_ros2.lane_node:main",
            "traffic_light_dataset_capture = vision_pkg_ros2.traffic_light_dataset_capture:main",
            "traffic_light_yolo_viewer = vision_pkg_ros2.traffic_light_yolo_viewer:main",
        ],
    },
)
