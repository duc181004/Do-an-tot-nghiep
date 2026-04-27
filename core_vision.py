import numpy as np
import math
from threading import Thread
import cv2

def calculate_mar(landmarks, w, h):
    p_left, p_right = landmarks.landmark[61], landmarks.landmark[291]
    p_top, p_bottom = landmarks.landmark[13], landmarks.landmark[14]
    dist_h = math.hypot((p_right.x - p_left.x)*w, (p_right.y - p_left.y)*h)
    dist_v = math.hypot((p_bottom.x - p_top.x)*w, (p_bottom.y - p_top.y)*h)
    return dist_v / dist_h if dist_h > 0 else 0.0

def calculate_ear(landmarks, idxs, w, h):
    p = [(landmarks.landmark[i].x * w, landmarks.landmark[i].y * h) for i in idxs]
    v1 = math.hypot(p[1][0] - p[5][0], p[1][1] - p[5][1])
    v2 = math.hypot(p[2][0] - p[4][0], p[2][1] - p[4][1])
    h_dist = math.hypot(p[0][0] - p[3][0], p[0][1] - p[3][1])
    return (v1 + v2) / (2.0 * h_dist) if h_dist > 0 else 0.0

def calculate_head_ratio(landmarks, w, h):
    n, c, f = landmarks.landmark[1], landmarks.landmark[152], landmarks.landmark[10]
    dist_nc = math.hypot((n.x - c.x)*w, (n.y - c.y)*h)
    dist_nf = math.hypot((n.x - f.x)*w, (n.y - f.y)*h)
    return dist_nc / dist_nf if dist_nf > 0 else 0.0

def get_crop(frame, coords, idxs, w, h, pad=0.2):
    pts = coords[idxs]
    x_min, y_min = np.min(pts, axis=0)
    x_max, y_max = np.max(pts, axis=0)
    pw, ph = int(pad * (x_max - x_min)), int(pad * (y_max - y_min))
    return frame[max(0, int(y_min)-ph):min(h, int(y_max)+ph), max(0, int(x_min)-pw):min(w, int(x_max)+pw)], (x_min, y_min, x_max, y_max)
