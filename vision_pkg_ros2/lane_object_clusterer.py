import cv2
import numpy as np

from . import config as cfg


class LaneObjectClusterer:
    """Clustering: white/yellow BEV masks -> lane object dictionaries."""

    def build(self, white_mask, yellow_mask):
        lane_objects = []
        max_count = int(getattr(cfg, "LANE_OBJECT_MAX_COUNT", 100))
        lane_objects.extend(
            self._build_from_mask(
                white_mask,
                int(getattr(cfg, "LANE_OBJECT_WHITE_LABEL", 2)),
                max_count,
            )
        )
        remaining = max(0, max_count - len(lane_objects))
        if remaining > 0:
            lane_objects.extend(
                self._build_from_mask(
                    yellow_mask,
                    int(getattr(cfg, "LANE_OBJECT_YELLOW_LABEL", 3)),
                    remaining,
                )
            )
        lane_objects.sort(key=lambda item: (float(item["x"]), abs(float(item["y"]))))
        return lane_objects[:max_count]

    def _build_from_mask(self, mask, label, max_count):
        if mask is None or max_count <= 0 or cv2.countNonZero(mask) == 0:
            return []

        h, w = mask.shape[:2]
        m_per_px = max(1e-6, float(getattr(cfg, "M_PER_PIXEL", 0.04)))
        segment_px = max(
            1,
            int(round(float(getattr(cfg, "LANE_OBJECT_LENGTH_M", 0.40)) / m_per_px)),
        )
        min_pixels = max(1, int(getattr(cfg, "LANE_OBJECT_MIN_PIXELS", 40)))
        cluster_gap = max(1, int(getattr(cfg, "LANE_OBJECT_CLUSTER_GAP_PX", 18)))
        clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        ys, xs = np.nonzero(clean > 0)
        if xs.size == 0:
            return []

        objects = []
        for y_top in range(0, h, segment_px):
            if len(objects) >= max_count:
                break
            y_bottom = min(h, y_top + segment_px)
            in_band = (ys >= y_top) & (ys < y_bottom)
            if not np.any(in_band):
                continue

            band_xs = xs[in_band]
            band_ys = ys[in_band]
            order = np.argsort(band_xs)
            band_xs = band_xs[order]
            band_ys = band_ys[order]

            split_points = np.where(np.diff(band_xs) > cluster_gap)[0] + 1
            x_groups = np.split(band_xs, split_points)
            y_groups = np.split(band_ys, split_points)
            for group_xs, group_ys in zip(x_groups, y_groups):
                if len(objects) >= max_count:
                    break
                if group_xs.size < min_pixels:
                    continue

                center_x = float(np.mean(group_xs))
                center_y = float(np.mean(group_ys))
                forward_m, lateral_m = self.bev_pixel_to_local_m(center_x, center_y, h, w)
                if forward_m < 0.0 or forward_m > float(getattr(cfg, "LANE_OBJECT_MAX_FORWARD_M", 10.0)):
                    continue
                objects.append(
                    {
                        "x": forward_m,
                        "y": lateral_m,
                        "x_size": float(getattr(cfg, "LANE_OBJECT_LENGTH_M", 0.40)),
                        "y_size": float(getattr(cfg, "LANE_OBJECT_WIDTH_M", 0.12)),
                        "yaw": self._estimate_yaw(group_xs, group_ys, h, w),
                        "label": int(label),
                    }
                )
        return objects

    @staticmethod
    def bev_pixel_to_local_m(px, py, h, w):
        m_per_px = float(getattr(cfg, "M_PER_PIXEL", 0.04))
        bottom_forward_px = float(getattr(cfg, "BEV_BOTTOM_FORWARD_M", 0.75)) / max(m_per_px, 1e-6)
        forward_px = float(h) + bottom_forward_px - float(py)
        lateral_px = float((w / 2.0) - px)
        return forward_px * m_per_px, lateral_px * m_per_px

    def _estimate_yaw(self, xs, ys, h, w):
        if xs.size < 2:
            return 0.0
        points = np.array(
            [self.bev_pixel_to_local_m(float(px), float(py), h, w) for px, py in zip(xs, ys)],
            dtype=np.float64,
        )
        points -= np.mean(points, axis=0)
        if np.max(np.abs(points)) < 1e-6:
            return 0.0
        _, _, vh = np.linalg.svd(points, full_matrices=False)
        direction = vh[0]
        if direction[0] < 0.0:
            direction = -direction
        return float(np.arctan2(direction[1], direction[0]))
