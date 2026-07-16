from . import config as cfg


class SchoolzoneDetector:
    """Detect a school zone from the number of yellow lane clusters."""

    def detect(self, lane_objects):
        yellow_label = int(getattr(cfg, "LANE_OBJECT_YELLOW_LABEL", 3))
        yellow_count = sum(
            1
            for obj in lane_objects or []
            if int(obj.get("label", -1)) == yellow_label
        )
        min_count = int(getattr(cfg, "SCHOOLZONE_MIN_YELLOW_CLUSTERS", 20))
        return yellow_count >= min_count
