from . import config as cfg


class TrafficLightDecider:
    """Traffic_Decide: detections/status -> integer command."""

    @staticmethod
    def select_status(detections):
        traffic_lights = [
            det for det in detections or []
            if TrafficLightDecider.is_traffic_light_name(det.get("name"))
        ]
        if not traffic_lights:
            return "없음"

        valid = [
            det for det in traffic_lights
            if det.get("signal", {}).get("state") not in (None, "없음")
        ]
        target = max(valid or traffic_lights, key=lambda det: float(det.get("conf", 0.0)))
        return target.get("signal", {}).get("state", "없음")

    @staticmethod
    def status_to_decision(status):
        # judgement_pkg TrafficLight mapping: 0=UNKNOWN, 1=GREEN, 2=LEFT, 3=RED.
        # Yellow is handled as RED so the vehicle keeps the stop behavior.
        return {
            "없음": int(getattr(cfg, "TRAFFIC_DECIDE_NONE", 0)),
            "정지": int(getattr(cfg, "TRAFFIC_DECIDE_STOP", 3)),
            "STOP": int(getattr(cfg, "TRAFFIC_DECIDE_STOP", 3)),
            "NONE": int(getattr(cfg, "TRAFFIC_DECIDE_NONE", 0)),
            "직진": int(getattr(cfg, "TRAFFIC_DECIDE_GO", 1)),
            "GO": int(getattr(cfg, "TRAFFIC_DECIDE_GO", 1)),
            "좌회전": int(getattr(cfg, "TRAFFIC_DECIDE_LEFT", 2)),
            "LEFT": int(getattr(cfg, "TRAFFIC_DECIDE_LEFT", 2)),
            "노랑": int(getattr(cfg, "TRAFFIC_DECIDE_YELLOW", 3)),
            "YELLOW": int(getattr(cfg, "TRAFFIC_DECIDE_YELLOW", 3)),
        }.get(status, int(getattr(cfg, "TRAFFIC_DECIDE_NONE", 0)))

    @staticmethod
    def status_to_ascii(status):
        return {
            "정지": "STOP",
            "직진": "GO",
            "좌회전": "LEFT",
            "노랑": "YELLOW",
            "없음": "NONE",
        }.get(status, "NONE")

    @staticmethod
    def normalize_yolo_name(name):
        return str(name).strip().replace("_", " ").lower()

    @staticmethod
    def is_traffic_light_name(name):
        return TrafficLightDecider.normalize_yolo_name(name) == "traffic light"

    @staticmethod
    def traffic_light_box_score(det):
        x1, y1, x2, y2 = det.get("bbox", (0.0, 0.0, 0.0, 0.0))
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        return area * max(0.0, float(det.get("conf", 0.0)))
