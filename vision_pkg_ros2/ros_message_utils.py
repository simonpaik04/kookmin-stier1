from interfaces.msg import Objects

from . import config as cfg


def create_lane_objects_msg(lane_objects, stamp, frame_id="stier"):
    objects_msg = Objects()
    objects_msg.header.stamp = stamp
    objects_msg.header.frame_id = frame_id

    max_count = min(100, int(getattr(cfg, "LANE_OBJECT_MAX_COUNT", 100)))
    selected = list(lane_objects or [])[:max_count]
    objects_msg.length = len(selected)

    objects_msg.x = [0.0] * 100
    objects_msg.y = [0.0] * 100
    objects_msg.x_size = [0.0] * 100
    objects_msg.y_size = [0.0] * 100
    objects_msg.yaw = [0.0] * 100
    objects_msg.lable = [0] * 100

    for idx, obj in enumerate(selected):
        objects_msg.x[idx] = float(obj.get("x", 0.0))
        objects_msg.y[idx] = float(obj.get("y", 0.0))
        objects_msg.x_size[idx] = float(obj.get("x_size", 0.0))
        objects_msg.y_size[idx] = float(obj.get("y_size", 0.0))
        objects_msg.yaw[idx] = float(obj.get("yaw", 0.0))
        objects_msg.lable[idx] = int(obj.get("label", 0))

    return objects_msg
