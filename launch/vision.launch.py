from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    image_topic = LaunchConfiguration("image_topic")
    compressed = LaunchConfiguration("compressed")
    yolo_model_path = LaunchConfiguration("yolo_model_path")
    show_rqt = LaunchConfiguration("show_rqt")
    show_params = LaunchConfiguration("show_params")
    view_topic = LaunchConfiguration("view_topic")

    return LaunchDescription([
        DeclareLaunchArgument(
            "image_topic",
            default_value="/usb_cam/image_raw/front",
            description="Input camera topic.",
        ),
        DeclareLaunchArgument(
            "compressed",
            default_value="false",
            description="Set true when image_topic is sensor_msgs/CompressedImage.",
        ),
        DeclareLaunchArgument(
            "yolo_model_path",
            default_value="",
            description="Optional YOLO model path. Empty string uses the package default model.",
        ),
        DeclareLaunchArgument(
            "show_rqt",
            default_value="true",
            description="Start rqt_image_view for the selected debug image topic.",
        ),
        DeclareLaunchArgument(
            "view_topic",
            default_value="/vision/front_yolo",
            description="Image topic opened by rqt_image_view.",
        ),
        DeclareLaunchArgument(
            "show_params",
            default_value="true",
            description="Start rqt_reconfigure so lane_node parameters can be tuned live.",
        ),
        Node(
            package="vision_pkg_ros2",
            executable="lane_node",
            name="lane_node",
            output="screen",
            parameters=[{
                "image_topic": image_topic,
                "compressed": compressed,
                "yolo_model_path": yolo_model_path,
            }],
        ),
        Node(
            package="rqt_image_view",
            executable="rqt_image_view",
            name="vision_result_view",
            arguments=[view_topic],
            condition=IfCondition(show_rqt),
            output="screen",
        ),
        Node(
            package="rqt_reconfigure",
            executable="rqt_reconfigure",
            name="vision_param_tuner",
            condition=IfCondition(show_params),
            output="screen",
        ),
    ])
