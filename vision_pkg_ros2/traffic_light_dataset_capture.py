import os
import time

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image

from . import config as cfg
from .traffic_light_crop import crop_top


class TrafficLightDatasetCapture(Node):
    def __init__(self):
        super().__init__("traffic_light_dataset_capture")
        self.declare_parameter("image_topic", cfg.IMAGE_TOPIC)
        self.declare_parameter("compressed", False)
        self.declare_parameter("output_dir", "/home/xytron/xycar_ws/datasets/sim_traffic_light")
        self.declare_parameter("split", "train")
        self.declare_parameter("save_fps", 0.5)
        self.declare_parameter("max_images", 30)
        self.declare_parameter("top_crop_ratio", cfg.YOLO_TRAFFIC_LIGHT_TOP_RATIO)
        self.declare_parameter("prefix", "sim_tl")

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.compressed = bool(self.get_parameter("compressed").value)
        self.output_dir = str(self.get_parameter("output_dir").value)
        self.split = str(self.get_parameter("split").value)
        self.save_fps = float(self.get_parameter("save_fps").value)
        self.max_images = int(self.get_parameter("max_images").value)
        self.top_crop_ratio = float(self.get_parameter("top_crop_ratio").value)
        self.prefix = str(self.get_parameter("prefix").value)

        self.bridge = CvBridge()
        self.count = 0
        self.capture_done = False
        self.last_save_time = 0.0

        self.images_dir = os.path.join(self.output_dir, "images", self.split)
        self.labels_dir = os.path.join(self.output_dir, "labels", self.split)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)
        self._write_dataset_yaml()

        msg_type = CompressedImage if self.compressed else Image
        self.sub = self.create_subscription(
            msg_type,
            self.image_topic,
            self._on_image,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            f"Capturing {self.image_topic} -> {self.images_dir}, "
            f"fps={self.save_fps}, max_images={self.max_images}, top_crop_ratio={self.top_crop_ratio}"
        )

    def _write_dataset_yaml(self):
        yaml_path = os.path.join(self.output_dir, "data.yaml")
        names = ["stop", "go", "left", "yellow"]
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(f"path: {self.output_dir}\n")
            f.write("train: images/train\n")
            f.write("val: images/val\n")
            f.write("names:\n")
            for idx, name in enumerate(names):
                f.write(f"  {idx}: {name}\n")

    def _on_image(self, msg):
        if self.max_images > 0 and self.count >= self.max_images:
            self.get_logger().info(f"Capture complete: {self.count} images saved")
            self.capture_done = True
            return

        now = time.time()
        interval = 0.0 if self.save_fps <= 0.0 else 1.0 / self.save_fps
        if now - self.last_save_time < interval:
            return

        try:
            if self.compressed:
                frame = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            else:
                frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as exc:
            self.get_logger().warning(f"Image conversion failed: {exc}")
            return

        frame, _ = crop_top(frame, self.top_crop_ratio)
        stamp = getattr(msg.header, "stamp", None)
        if stamp is not None:
            stamp_text = f"{int(stamp.sec)}_{int(stamp.nanosec):09d}"
        else:
            stamp_text = f"{int(now)}_{int((now % 1.0) * 1e9):09d}"
        stem = f"{self.prefix}_{stamp_text}_{self.count:06d}"
        image_path = os.path.join(self.images_dir, stem + ".jpg")
        label_path = os.path.join(self.labels_dir, stem + ".txt")

        if not cv2.imwrite(image_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
            self.get_logger().warning(f"Failed to write image: {image_path}")
            return
        open(label_path, "a", encoding="utf-8").close()

        self.count += 1
        self.last_save_time = now
        if self.count == 1 or self.count % 25 == 0:
            self.get_logger().info(f"Saved {self.count}/{self.max_images}: {image_path}")

def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightDatasetCapture()
    try:
        while rclpy.ok() and not node.capture_done:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
