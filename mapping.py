# mapping.py
import os
import math
import time
import numpy as np
import cv2

def _sigmoid(lo: float) -> float:
    return 1.0 / (1.0 + math.exp(-lo))

def render_occupancy_map(mem, pose, scale: int = 6, pad: int = 6, draw_grid: bool = False):
    """
    Returns a BGR image for cv2.imshow.
    Unknown = gray
    Free    = light
    Occ     = dark
    Visited = green overlay
    Robot   = red + heading arrow
    """
    # Collect all known cells so we can build bounds
    cells = set(mem.logodds.keys()) | set(mem.visited.keys())
    rx, ry = mem._cell(pose.x_cm, pose.y_cm)
    cells.add((rx, ry))

    if not cells:
        img = np.full((200, 200, 3), 127, np.uint8)
        return cv2.resize(img, (200 * scale, 200 * scale), interpolation=cv2.INTER_NEAREST)

    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)

    w = (maxx - minx + 1) + 2 * pad
    h = (maxy - miny + 1) + 2 * pad

    # Base: unknown
    grid = np.full((h, w, 3), 127, np.uint8)

    # Occupancy shading
    for (ix, iy), lo in mem.logodds.items():
        gx = (ix - minx) + pad
        gy = (iy - miny) + pad
        p = _sigmoid(lo)

        if p > 0.65:      # occupied
            grid[gy, gx] = (15, 15, 15)
        elif p < 0.35:    # free
            grid[gy, gx] = (235, 235, 235)
        else:             # unknown-ish
            grid[gy, gx] = (127, 127, 127)

    # Visited overlay
    for (ix, iy), count in mem.visited.items():
        gx = (ix - minx) + pad
        gy = (iy - miny) + pad
        if 0 <= gx < w and 0 <= gy < h:
            g = min(255, 40 + count * 25)
            # Blend a green overlay
            base = grid[gy, gx].astype(np.float32)
            overlay = np.array([0, g, 0], np.float32)
            grid[gy, gx] = np.clip(base * 0.55 + overlay * 0.45, 0, 255).astype(np.uint8)

    # Robot marker
    rpx = (rx - minx) + pad
    rpy = (ry - miny) + pad
    if 0 <= rpx < w and 0 <= rpy < h:
        grid[rpy, rpx] = (0, 0, 255)  # red (BGR)

    # Upscale for display
    img = cv2.resize(grid, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

    # Optional grid lines
    if draw_grid:
        for x in range(0, img.shape[1], scale):
            cv2.line(img, (x, 0), (x, img.shape[0]), (60, 60, 60), 1)
        for y in range(0, img.shape[0], scale):
            cv2.line(img, (0, y), (img.shape[1], y), (60, 60, 60), 1)

    # Heading arrow
    cx = rpx * scale + scale // 2
    cy = rpy * scale + scale // 2
    th = math.radians(pose.th_deg)
    ax = int(cx + math.cos(th) * scale * 4)
    ay = int(cy + math.sin(th) * scale * 4)
    cv2.arrowedLine(img, (cx, cy), (ax, ay), (0, 0, 255), 2, tipLength=0.35)

    return img

def save_map_png(path: str, img_bgr) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img_bgr)
