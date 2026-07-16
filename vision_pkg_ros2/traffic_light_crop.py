from . import config as cfg


def crop_top(frame, ratio=None):
    """Crop: return the upper image region used for traffic-light YOLO."""
    if frame is None:
        return None, 0

    if ratio is None:
        ratio = float(getattr(cfg, "YOLO_TRAFFIC_LIGHT_TOP_RATIO", 0.60))

    h = frame.shape[0]
    crop_h = int(max(1, min(h, h * float(ratio))))
    return frame[:crop_h, :], crop_h
