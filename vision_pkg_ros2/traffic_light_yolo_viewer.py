import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from ultralytics import YOLO

from . import config as cfg
from .traffic_light_classifier import TrafficLightColorClassifier
from .traffic_light_crop import crop_top
from .traffic_light_decider import TrafficLightDecider


class TrafficLightYoloViewer(Node):
    def __init__(self):
        super().__init__("traffic_light_yolo_viewer")
        self.declare_parameter("image_topic", cfg.IMAGE_TOPIC)
        self.declare_parameter("compressed", False)
        self.declare_parameter("model_path", cfg.YOLO_MODEL_PATH)
        self.declare_parameter("conf", cfg.YOLO_TRAFFIC_LIGHT_CONF)
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("top_ratio", cfg.YOLO_TRAFFIC_LIGHT_TOP_RATIO)

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.compressed = bool(self.get_parameter("compressed").value)
        self.model_path = str(self.get_parameter("model_path").value)
        self.conf = float(self.get_parameter("conf").value)
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.top_ratio = float(self.get_parameter("top_ratio").value)

        self.bridge = CvBridge()
        self.classifier = TrafficLightColorClassifier()
        self.model = YOLO(self.model_path)
        self.names = getattr(self.model, "names", {}) or {}
        self.traffic_light_ids = [
            int(cls_id)
            for cls_id, name in dict(self.names).items()
            if TrafficLightDecider.is_traffic_light_name(name)
        ]

        msg_type = CompressedImage if self.compressed else Image
        self.sub = self.create_subscription(msg_type, self.image_topic, self._on_image, 10)
        self.get_logger().info(
            f"Traffic light YOLO viewer started. topic={self.image_topic}, compressed={self.compressed}, "
            f"model={self.model_path}, conf={self.conf}, top_ratio={self.top_ratio}"
        )

    def _on_image(self, msg):
        try:
            if self.compressed:
                frame = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            else:
                frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as exc:
            self.get_logger().warn(f"image convert failed: {exc}")
            return

        view = frame.copy()
        w = frame.shape[1]
        crop, crop_h = crop_top(frame, self.top_ratio)
        cv2.line(view, (0, crop_h), (w - 1, crop_h), (255, 255, 0), 1)

        status = "NONE"
        try:
            results = self.model.predict(
                crop,
                imgsz=self.imgsz,
                conf=self.conf,
                classes=self.traffic_light_ids if self.traffic_light_ids else None,
                verbose=False,
            )
        except Exception as exc:
            self.get_logger().warn(f"YOLO predict failed: {exc}")
            results = []

        boxes = getattr(results[0], "boxes", None) if results else None
        detections = []
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0])
                name = TrafficLightDecider.normalize_yolo_name(self.names.get(cls_id, cls_id))
                if not TrafficLightDecider.is_traffic_light_name(name):
                    continue

                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy().astype(float).tolist()
                bbox = (x1, y1, x2, y2)
                signal = self.classifier.classify(frame, bbox)
                state_en = signal.get("state_en", "NONE")
                if state_en != "NONE":
                    status = state_en
                detections.append({"conf": conf, "bbox": bbox, "state_en": state_en})

        if detections:
            det = max(detections, key=TrafficLightDecider.traffic_light_box_score)
            x1, y1, x2, y2 = det["bbox"]
            conf = det["conf"]
            state_en = det["state_en"]
            p1 = (int(round(x1)), int(round(y1)))
            p2 = (int(round(x2)), int(round(y2)))
            color = self._color_for_state(state_en)
            cv2.rectangle(view, p1, p2, color, 2)
            cv2.putText(
                view,
                f"traffic_light {conf:.2f} {state_en}",
                (p1[0], max(18, p1[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            view,
            f"TRAFFIC: {status}",
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            self._color_for_state(status),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("trained_yolo_traffic_light", view)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            rclpy.shutdown()

    @staticmethod
    def _color_for_state(state):
        return {
            "STOP": (0, 0, 255),
            "YELLOW": (0, 255, 255),
            "GO": (0, 220, 0),
            "LEFT": (0, 220, 0),
        }.get(state, (220, 220, 220))


def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightYoloViewer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
