from . import config as cfg
from .traffic_light_classifier import TrafficLightColorClassifier
from .traffic_light_crop import crop_top
from .traffic_light_decider import TrafficLightDecider
from .traffic_light_yolo import YoloNet


class TrafficLightProcessor:
    """Traffic Light Process: Crop -> YOLO_net -> color classifier -> decide."""

    def __init__(self, logger=None):
        self.yolo = YoloNet(logger=logger)
        self.classifier = TrafficLightColorClassifier()
        self.frame_idx = 0
        self.crop_idx = 0
        self.last_detections = []
        self.last_crop_detections = []
        self.last_yolo_count = 0
        self.traffic_light_status = "없음"
        self.last_traffic_decide = int(getattr(cfg, "TRAFFIC_DECIDE_NONE", 4))

    def process(self, frame):
        detections = self._run_yolo(frame)
        self.traffic_light_status = TrafficLightDecider.select_status(detections)
        self.last_traffic_decide = TrafficLightDecider.status_to_decision(self.traffic_light_status)
        return {
            "detections": detections,
            "traffic_light_status": self.traffic_light_status,
            "traffic_decide": self.last_traffic_decide,
        }

    def _run_yolo(self, frame):
        if not getattr(cfg, "YOLO_ENABLED", True) or frame is None:
            self._clear_yolo_state()
            return []

        self.frame_idx += 1
        skip = max(1, int(getattr(cfg, "YOLO_FRAME_SKIP", 1)))
        if skip > 1 and (self.frame_idx % skip) != 1:
            return self.last_detections

        if self.yolo.load() is None:
            self._clear_yolo_state()
            return []

        detections = self._detect_traffic_lights_top_crop(frame)

        self.last_detections = detections
        self.last_yolo_count = len(detections)
        return detections

    def _detect_traffic_lights_top_crop(self, frame):
        self.crop_idx += 1
        skip = max(1, int(getattr(cfg, "YOLO_TRAFFIC_LIGHT_FRAME_SKIP", 4)))
        if skip > 1 and (self.crop_idx % skip) != 1:
            return list(self.last_crop_detections)

        crop, _ = crop_top(frame)
        results = self.yolo.predict(
            crop,
            imgsz=int(getattr(cfg, "YOLO_TRAFFIC_LIGHT_IMGSZ", 640)),
            conf=float(getattr(cfg, "YOLO_TRAFFIC_LIGHT_CONF", 0.18)),
            classes=self._traffic_light_class_ids() or None,
        )
        boxes = getattr(results[0], "boxes", None) if results else None
        if boxes is None:
            self.last_crop_detections = []
            return []

        found = []
        for box in boxes:
            cls_id = int(box.cls[0])
            name = TrafficLightDecider.normalize_yolo_name(self.yolo.names.get(cls_id, cls_id))
            if not TrafficLightDecider.is_traffic_light_name(name):
                continue
            x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy().astype(float).tolist()
            bbox = (x1, y1, x2, y2)
            signal = self.classifier.classify(frame, bbox)
            found.append(
                {
                    "name": "traffic light",
                    "conf": float(box.conf[0]),
                    "bbox": bbox,
                    "signal": signal,
                }
            )

        if len(found) > 1:
            found = [max(found, key=TrafficLightDecider.traffic_light_box_score)]
        self.last_crop_detections = found
        return found

    def _traffic_light_class_ids(self):
        ids = []
        for cls_id, name in dict(self.yolo.names).items():
            if TrafficLightDecider.is_traffic_light_name(name):
                ids.append(int(cls_id))
        return ids

    def _clear_yolo_state(self):
        self.last_detections = []
        self.last_crop_detections = []
        self.last_yolo_count = 0
        self.traffic_light_status = "없음"
