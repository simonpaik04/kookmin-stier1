import cv2
import numpy as np
import rclpy
import time
from cv_bridge import CvBridge
from interfaces.msg import Objects
from rcl_interfaces.msg import FloatingPointRange, IntegerRange, ParameterDescriptor, SetParametersResult
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool, Float32, Int32

from . import config as cfg
from . import ros_message_utils
from .lane_processor import LaneProcessor


class LaneNode(Node):
    def __init__(self):
        super().__init__("lane_node")

        self.bridge = CvBridge()
        self.processor = LaneProcessor(logger=self.get_logger(), clock=self.get_clock())
        self._last_process_time = 0.0

        self._declare_parameters()
        self._load_parameters()
        self.add_on_set_parameters_callback(self._on_parameter_update)

        self.pub_lane_mask = self.create_publisher(Image, "/vision/lane_mask", 1)
        self.pub_bev = self.create_publisher(Image, "/vision/bev", 1)
        self.pub_final = self.create_publisher(Image, "/vision/final_result", 1)
        self.pub_front_yolo = self.create_publisher(Image, cfg.FRONT_YOLO_TOPIC, 1)

        self.pub_lane_objects = self.create_publisher(
            Objects,
            cfg.LANE_OBJECT_TOPIC,
            1,
        )
        self.pub_schoolzone_detect = self.create_publisher(
            Bool,
            cfg.SCHOOLZONE_DETECT_TOPIC,
            1,
        )
        self.pub_stopline_detect = self.create_publisher(
            Bool,
            cfg.STOPLINE_DETECT_TOPIC,
            1,
        )
        self.pub_stopline_distance = self.create_publisher(
            Float32,
            cfg.STOPLINE_DISTANCE_TOPIC,
            1,
        )
        self.pub_traffic_decide = self.create_publisher(
            Int32,
            cfg.TRAFFIC_DECIDE_TOPIC,
            1,
        )

        msg_type = CompressedImage if self.compressed else Image
        self.sub = self.create_subscription(
            msg_type,
            self.image_topic,
            self.callback,
            1,
        )

        self.get_logger().info(
            f"Lane node started. image_topic={self.image_topic}, compressed={self.compressed}"
        )

    def _declare_parameters(self):
        defaults = {
            "image_topic": cfg.IMAGE_TOPIC,
            "compressed": False,
            "mode": "sim",
            "publish_debug_images": cfg.PUBLISH_DEBUG_IMAGES,
            "target_fps": cfg.TARGET_FPS,
            "bev_length": cfg.BEV_LENGTH_M,
            "bev_width": cfg.BEV_WIDTH_M,
            "bev_tuning_enabled": cfg.BEV_TUNING_ENABLED,
            "bev_vehicle_length": cfg.BEV_VEHICLE_LENGTH_M,
            "bev_vehicle_width": cfg.BEV_VEHICLE_WIDTH_M,
            "bev_bottom_forward_m": cfg.BEV_BOTTOM_FORWARD_M,
            "bev_lane_width": cfg.BEV_LANE_WIDTH_M,
            "bev_source_cutoff_y": cfg.BEV_SOURCE_CUTOFF_Y,
            "bev_src_top_left_x": cfg.BEV_SRC_TOP_LEFT_X,
            "bev_src_top_right_x": cfg.BEV_SRC_TOP_RIGHT_X,
            "bev_src_top_y": cfg.BEV_SRC_TOP_Y,
            "bev_src_bottom_left_x": cfg.BEV_SRC_BOTTOM_LEFT_X,
            "bev_src_bottom_right_x": cfg.BEV_SRC_BOTTOM_RIGHT_X,
            "bev_src_bottom_y": cfg.BEV_SRC_BOTTOM_Y,
            "ignore_center_yellow": cfg.IGNORE_CENTER_YELLOW,
            "center_yellow_ignore_width_px": cfg.CENTER_YELLOW_IGNORE_WIDTH_PX,
            "stop_line_min_width_px": cfg.STOP_LINE_MIN_WIDTH_PX,
            "stop_line_min_height_px": cfg.STOP_LINE_MIN_HEIGHT_PX,
            "stop_line_row_ratio": cfg.STOP_LINE_ROW_RATIO,
            "stop_line_clear_margin_px": cfg.STOP_LINE_CLEAR_MARGIN_PX,
            "stop_line_trigger_distance_m": cfg.STOP_LINE_TRIGGER_DISTANCE_M,
            "lane_object_max_count": cfg.LANE_OBJECT_MAX_COUNT,
            "lane_object_max_forward_m": cfg.LANE_OBJECT_MAX_FORWARD_M,
            "lane_object_length_m": cfg.LANE_OBJECT_LENGTH_M,
            "lane_object_width_m": cfg.LANE_OBJECT_WIDTH_M,
            "lane_object_min_pixels": cfg.LANE_OBJECT_MIN_PIXELS,
            "lane_object_cluster_gap_px": cfg.LANE_OBJECT_CLUSTER_GAP_PX,
            "schoolzone_min_yellow_clusters": cfg.SCHOOLZONE_MIN_YELLOW_CLUSTERS,
            "yolo_enabled": cfg.YOLO_ENABLED,
            "yolo_model_path": cfg.YOLO_MODEL_PATH,
            "yolo_frame_skip": cfg.YOLO_FRAME_SKIP,
            "yolo_traffic_light_conf": cfg.YOLO_TRAFFIC_LIGHT_CONF,
            "yolo_traffic_light_top_ratio": cfg.YOLO_TRAFFIC_LIGHT_TOP_RATIO,
            "yolo_traffic_light_frame_skip": cfg.YOLO_TRAFFIC_LIGHT_FRAME_SKIP,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value, self._descriptor_for(name))

    @staticmethod
    def _descriptor_for(name):
        int_ranges = {
            "center_yellow_ignore_width_px": (0, 500, 1),
            "stop_line_min_width_px": (10, 1000, 1),
            "stop_line_min_height_px": (1, 100, 1),
            "stop_line_clear_margin_px": (0, 100, 1),
            "lane_object_max_count": (1, 100, 1),
            "lane_object_min_pixels": (1, 500, 1),
            "lane_object_cluster_gap_px": (1, 200, 1),
            "schoolzone_min_yellow_clusters": (1, 100, 1),
            "yolo_frame_skip": (1, 30, 1),
            "yolo_traffic_light_frame_skip": (1, 30, 1),
            "bev_source_cutoff_y": (0, 480, 1),
            "bev_src_top_left_x": (0, 640, 1),
            "bev_src_top_right_x": (0, 640, 1),
            "bev_src_top_y": (0, 480, 1),
            "bev_src_bottom_left_x": (0, 640, 1),
            "bev_src_bottom_right_x": (0, 640, 1),
            "bev_src_bottom_y": (0, 480, 1),
        }
        float_ranges = {
            "bev_length": (5.0, 40.0, 0.1),
            "bev_width": (2.0, 20.0, 0.1),
            "bev_vehicle_length": (1.0, 10.0, 0.1),
            "bev_vehicle_width": (0.5, 5.0, 0.1),
            "bev_bottom_forward_m": (0.0, 5.0, 0.05),
            "bev_lane_width": (1.0, 12.0, 0.1),
            "target_fps": (0.0, 30.0, 1.0),
            "stop_line_row_ratio": (0.05, 1.0, 0.01),
            "stop_line_trigger_distance_m": (0.0, 30.0, 0.1),
            "lane_object_max_forward_m": (1.0, 30.0, 0.05),
            "lane_object_length_m": (0.05, 2.0, 0.01),
            "lane_object_width_m": (0.02, 1.0, 0.01),
            "yolo_traffic_light_conf": (0.05, 0.95, 0.01),
            "yolo_traffic_light_top_ratio": (0.1, 1.0, 0.01),
        }
        descriptor = ParameterDescriptor(description=f"lane_node runtime parameter: {name}")
        if name in int_ranges:
            start, end, step = int_ranges[name]
            descriptor.integer_range = [IntegerRange(from_value=start, to_value=end, step=step)]
        elif name in float_ranges:
            start, end, step = float_ranges[name]
            descriptor.floating_point_range = [
                FloatingPointRange(from_value=start, to_value=end, step=step)
            ]
        return descriptor

    def _load_parameters(self):
        self.image_topic = self.get_parameter("image_topic").value
        self.compressed = self._as_bool(self.get_parameter("compressed").value)

        mode = self.get_parameter("mode").value
        if mode == "real":
            cfg.K = cfg.K_REAL
            cfg.DIST_COEFF = cfg.DIST_COEFF_REAL
        else:
            cfg.K = None
            cfg.DIST_COEFF = np.zeros(5, dtype=np.float32)

        self._apply_tunable_parameters()

    @staticmethod
    def _as_bool(value):
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "on"}
        return bool(value)

    @staticmethod
    def _bev_parameter_names():
        return {
            "bev_tuning_enabled",
            "bev_length",
            "bev_width",
            "bev_vehicle_length",
            "bev_vehicle_width",
            "bev_bottom_forward_m",
            "bev_lane_width",
            "bev_source_cutoff_y",
            "bev_src_top_left_x",
            "bev_src_top_right_x",
            "bev_src_top_y",
            "bev_src_bottom_left_x",
            "bev_src_bottom_right_x",
            "bev_src_bottom_y",
        }

    def _apply_tunable_parameters(self):
        cfg.PUBLISH_DEBUG_IMAGES = self._as_bool(self.get_parameter("publish_debug_images").value)
        cfg.TARGET_FPS = min(30.0, max(0.0, float(self.get_parameter("target_fps").value)))
        cfg.BEV_LENGTH_M = float(self.get_parameter("bev_length").value)
        cfg.BEV_WIDTH_M = float(self.get_parameter("bev_width").value)
        cfg.BEV_TUNING_ENABLED = self._as_bool(self.get_parameter("bev_tuning_enabled").value)
        cfg.BEV_VEHICLE_LENGTH_M = float(self.get_parameter("bev_vehicle_length").value)
        cfg.BEV_VEHICLE_WIDTH_M = float(self.get_parameter("bev_vehicle_width").value)
        cfg.BEV_BOTTOM_FORWARD_M = float(self.get_parameter("bev_bottom_forward_m").value)
        cfg.BEV_LANE_WIDTH_M = float(self.get_parameter("bev_lane_width").value)
        cfg.BEV_SOURCE_CUTOFF_Y = int(self.get_parameter("bev_source_cutoff_y").value)
        cfg.BEV_SRC_TOP_LEFT_X = int(self.get_parameter("bev_src_top_left_x").value)
        cfg.BEV_SRC_TOP_RIGHT_X = int(self.get_parameter("bev_src_top_right_x").value)
        cfg.BEV_SRC_TOP_Y = int(self.get_parameter("bev_src_top_y").value)
        cfg.BEV_SRC_BOTTOM_LEFT_X = int(self.get_parameter("bev_src_bottom_left_x").value)
        cfg.BEV_SRC_BOTTOM_RIGHT_X = int(self.get_parameter("bev_src_bottom_right_x").value)
        cfg.BEV_SRC_BOTTOM_Y = int(self.get_parameter("bev_src_bottom_y").value)
        cfg.IGNORE_CENTER_YELLOW = self._as_bool(self.get_parameter("ignore_center_yellow").value)
        cfg.CENTER_YELLOW_IGNORE_WIDTH_PX = int(self.get_parameter("center_yellow_ignore_width_px").value)
        cfg.STOP_LINE_MIN_WIDTH_PX = int(self.get_parameter("stop_line_min_width_px").value)
        cfg.STOP_LINE_MIN_HEIGHT_PX = int(self.get_parameter("stop_line_min_height_px").value)
        cfg.STOP_LINE_ROW_RATIO = float(self.get_parameter("stop_line_row_ratio").value)
        cfg.STOP_LINE_CLEAR_MARGIN_PX = int(self.get_parameter("stop_line_clear_margin_px").value)
        cfg.STOP_LINE_TRIGGER_DISTANCE_M = float(
            self.get_parameter("stop_line_trigger_distance_m").value
        )
        cfg.LANE_OBJECT_MAX_COUNT = int(self.get_parameter("lane_object_max_count").value)
        cfg.LANE_OBJECT_MAX_FORWARD_M = float(self.get_parameter("lane_object_max_forward_m").value)
        cfg.LANE_OBJECT_LENGTH_M = float(self.get_parameter("lane_object_length_m").value)
        cfg.LANE_OBJECT_WIDTH_M = float(self.get_parameter("lane_object_width_m").value)
        cfg.LANE_OBJECT_MIN_PIXELS = int(self.get_parameter("lane_object_min_pixels").value)
        cfg.LANE_OBJECT_CLUSTER_GAP_PX = int(self.get_parameter("lane_object_cluster_gap_px").value)
        cfg.SCHOOLZONE_MIN_YELLOW_CLUSTERS = int(
            self.get_parameter("schoolzone_min_yellow_clusters").value
        )
        cfg.YOLO_ENABLED = self._as_bool(self.get_parameter("yolo_enabled").value)
        yolo_model_path = str(self.get_parameter("yolo_model_path").value).strip()
        if yolo_model_path:
            cfg.YOLO_MODEL_PATH = yolo_model_path
        cfg.YOLO_FRAME_SKIP = int(self.get_parameter("yolo_frame_skip").value)
        cfg.YOLO_TRAFFIC_LIGHT_CONF = float(self.get_parameter("yolo_traffic_light_conf").value)
        cfg.YOLO_TRAFFIC_LIGHT_TOP_RATIO = float(self.get_parameter("yolo_traffic_light_top_ratio").value)
        cfg.YOLO_TRAFFIC_LIGHT_FRAME_SKIP = int(self.get_parameter("yolo_traffic_light_frame_skip").value)

    def _on_parameter_update(self, params):
        for param in params:
            if param.name in {"image_topic", "compressed"}:
                return SetParametersResult(
                    successful=False,
                    reason="image_topic/compressed require node restart",
                )
        rebuild_bev = any(param.name in self._bev_parameter_names() for param in params)
        self._apply_tunable_parameters()
        if rebuild_bev:
            self.processor.rebuild_bev_projector()
        return SetParametersResult(successful=True)

    def callback(self, msg):
        target_fps = float(getattr(cfg, 'TARGET_FPS', 0.0))
        if target_fps > 0.0:
            now = time.monotonic()
            min_interval = 1.0 / target_fps
            if now - self._last_process_time < min_interval:
                return
            self._last_process_time = now

        try:
            if self.compressed:
                np_arr = np.frombuffer(msg.data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"Image conversion failed: {exc}")
            return

        if frame is None:
            return

        results = self.processor.process(frame)

        if cfg.PUBLISH_DEBUG_IMAGES:
            if results.get("bev") is not None:
                bev_encoding = "bgr8" if results["bev"].ndim == 3 else "mono8"
                self.pub_bev.publish(self.bridge.cv2_to_imgmsg(results["bev"], bev_encoding))
            if results.get("final") is not None:
                self.pub_final.publish(self.bridge.cv2_to_imgmsg(results["final"], "bgr8"))
            if results.get("front_yolo") is not None:
                front_yolo_msg = self.bridge.cv2_to_imgmsg(results["front_yolo"], "bgr8")
                front_yolo_msg.header = msg.header
                self.pub_front_yolo.publish(front_yolo_msg)
            if results.get("lane_mask") is not None:
                self.pub_lane_mask.publish(self.bridge.cv2_to_imgmsg(results["lane_mask"], "mono8"))

        stamp = self.get_clock().now().to_msg()
        self.pub_lane_objects.publish(
            ros_message_utils.create_lane_objects_msg(results.get("lane_objects"), stamp)
        )
        self.pub_schoolzone_detect.publish(
            Bool(data=bool(results.get("schoolzone_detect", False)))
        )
        self.pub_stopline_detect.publish(
            Bool(data=bool(results.get("stopline_detect", False)))
        )
        self.pub_stopline_distance.publish(
            Float32(data=float(results.get("stopline_distance", -1.0)))
        )
        self.pub_traffic_decide.publish(
            Int32(data=int(results.get("traffic_decide", cfg.TRAFFIC_DECIDE_NONE)))
        )

def main(args=None):
    rclpy.init(args=args)
    node = LaneNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
