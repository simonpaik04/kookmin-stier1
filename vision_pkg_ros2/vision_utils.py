import cv2
import numpy as np

from . import config as cfg

# ============================================================
# Preprocessing
# ============================================================

def white_yellow_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    white = cv2.inRange(hsv, (0, 0, 180), (179, 70, 255))
    yellow = cv2.inRange(hsv, (15, 70, 80), (40, 255, 255))
    return cv2.bitwise_or(white, yellow)

def split_white_yellow_masks(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 180), (179, 70, 255))
    yellow = cv2.inRange(hsv, (15, 70, 80), (40, 255, 255))
    return white, yellow

def keep_lane_colors(frame):
    return cv2.bitwise_and(frame, frame, mask=white_yellow_mask(frame))


def find_largest_black_box(frame):
    """Find the largest internal near-black rectangular area in a BEV image."""
    if frame is None or not getattr(cfg, "START_BLACK_BOX_FILTER_ENABLED", True):
        return None

    h, w = frame.shape[:2]
    if h == 0 or w == 0:
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    max_value = int(getattr(cfg, "START_BLACK_BOX_MAX_VALUE", 35))
    black_mask = cv2.inRange(gray, 0, max_value)

    border = max(
        1,
        int(round(min(h, w) * float(getattr(cfg, "START_BLACK_BOX_BORDER_MARGIN_RATIO", 0.03)))),
    )
    black_mask[:border, :] = 0
    black_mask[h - border:, :] = 0
    black_mask[:, :border] = 0
    black_mask[:, w - border:] = 0

    close_size = max(3, int(getattr(cfg, "START_BLACK_BOX_CLOSE_KERNEL_PX", 31)))
    if close_size % 2 == 0:
        close_size += 1
    closed = cv2.morphologyEx(
        black_mask,
        cv2.MORPH_CLOSE,
        np.ones((close_size, close_size), np.uint8),
    )

    open_size = max(1, int(getattr(cfg, "START_BLACK_BOX_OPEN_KERNEL_PX", 5)))
    if open_size > 1:
        closed = cv2.morphologyEx(
            closed,
            cv2.MORPH_OPEN,
            np.ones((open_size, open_size), np.uint8),
        )

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    min_width = w * float(getattr(cfg, "START_BLACK_BOX_MIN_WIDTH_RATIO", 0.40))
    min_height = h * float(getattr(cfg, "START_BLACK_BOX_MIN_HEIGHT_RATIO", 0.05))
    max_height = h * float(getattr(cfg, "START_BLACK_BOX_MAX_HEIGHT_RATIO", 0.45))
    min_area = h * w * float(getattr(cfg, "START_BLACK_BOX_MIN_AREA_RATIO", 0.02))
    min_fill = float(getattr(cfg, "START_BLACK_BOX_MIN_FILL_RATIO", 0.25))

    candidates = []
    for label_idx in range(1, n_labels):
        x, y, width, height, area = [int(value) for value in stats[label_idx]]
        if width < min_width or not (min_height <= height <= max_height):
            continue
        if area < min_area:
            continue
        original_fill = cv2.countNonZero(black_mask[y:y + height, x:x + width]) / max(
            float(width * height),
            1.0,
        )
        if original_fill < min_fill:
            continue
        candidates.append((width * height, x, y, x + width, y + height))

    if not candidates:
        return None
    _, x1, y1, x2, y2 = max(candidates, key=lambda item: item[0])
    return (x1, y1, x2, y2)


def black_out_box(image, box):
    """Return a copy with the selected rectangle filled with black."""
    if image is None:
        return None
    output = image.copy()
    if box is None:
        return output
    h, w = output.shape[:2]
    x1, y1, x2, y2 = box
    output[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 0
    return output

def detect_stop_line(white_mask):
    if white_mask is None:
        return None

    h, w = white_mask.shape[:2]
    if h == 0 or w == 0:
        return None

    kernel_w = max(15, int(w * 0.12))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 3))
    horizontal = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)

    row_counts = np.count_nonzero(horizontal, axis=1)
    min_row_count = max(int(cfg.STOP_LINE_MIN_WIDTH_PX), int(w * float(cfg.STOP_LINE_ROW_RATIO)))
    rows = np.where(row_counts >= min_row_count)[0]
    if rows.size == 0:
        return None

    bands = []
    start = int(rows[0])
    prev = int(rows[0])
    for row in rows[1:]:
        row = int(row)
        if row == prev + 1:
            prev = row
        else:
            bands.append((start, prev))
            start = prev = row
    bands.append((start, prev))

    min_h = int(cfg.STOP_LINE_MIN_HEIGHT_PX)
    bands = [band for band in bands if (band[1] - band[0] + 1) >= min_h]
    if not bands:
        return None

    y1, y2 = max(bands, key=lambda band: band[1])
    band_mask = horizontal[y1:y2 + 1, :]
    xs = np.where(np.count_nonzero(band_mask, axis=0) > 0)[0]
    if xs.size == 0:
        return None
    x1 = int(xs.min())
    x2 = int(xs.max())
    if (x2 - x1 + 1) < int(cfg.STOP_LINE_MIN_WIDTH_PX):
        return None
    return (x1, int(y1), x2, int(y2))


def stop_line_distance_m(stop_line, bev_height):
    """Return the nearest longitudinal distance to a BEV stop-line band."""
    if stop_line is None or bev_height <= 0:
        return -1.0

    _, _, _, nearest_y = stop_line
    m_per_px = max(1e-6, float(getattr(cfg, "M_PER_PIXEL", 0.04)))
    bottom_forward_px = float(getattr(cfg, "BEV_BOTTOM_FORWARD_M", 0.75)) / m_per_px
    forward_px = float(bev_height) + bottom_forward_px - float(nearest_y)
    return max(0.0, forward_px * m_per_px)

def remove_stop_line_from_masks(stop_line, masks, margin_px=None):
    if stop_line is None:
        return masks

    if margin_px is None:
        margin_px = getattr(cfg, 'STOP_LINE_CLEAR_MARGIN_PX', 14)

    _, y1, _, y2 = stop_line
    cleaned = []
    for mask in masks:
        if mask is None:
            cleaned.append(None)
            continue
        out = mask.copy()
        h, _ = out.shape[:2]
        half_h = max(
            int(margin_px),
            int(getattr(cfg, 'STOP_LINE_MIN_HEIGHT_PX', 8)),
        )
        clear_y1 = max(0, int(y1) - half_h)
        clear_y2 = min(h, int(y2) + half_h + 1)
        out[clear_y1:clear_y2, :] = 0
        cleaned.append(out)
    return tuple(cleaned)
