#!/usr/bin/env python3
"""
Flekks Movement Visualization (v6 — Overlapping Limb Waveforms)
- Per-limb movement as independent overlapping semi-transparent areas
- Lines cross each other freely — whichever limb is most active rises highest
- Minimal skeleton with brand colors
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import sys

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# Flekks brand accent (BGR)
ACCENT       = (166, 180, 64)
ACCENT_DIM   = (115, 125, 45)
WHITE_CORE   = (240, 245, 240)

# Per-region colors (BGR)
REGION_COLORS = {
    'torso':     (120, 130, 140),
    'left_arm':  (166, 180, 64),
    'right_arm': (219, 135, 71),
    'left_leg':  (204, 104, 132),
    'right_leg': (58, 112, 196),
}

REGION_LABELS = ['torso', 'left_arm', 'right_arm', 'left_leg', 'right_leg']
REGION_DISPLAY = {
    'torso': 'TORSO',
    'left_arm': 'L ARM',
    'right_arm': 'R ARM',
    'left_leg': 'L LEG',
    'right_leg': 'R LEG',
}

REGION_LANDMARKS = {
    'torso':     [11, 12, 23, 24],
    'left_arm':  [11, 13, 15],
    'right_arm': [12, 14, 16],
    'left_leg':  [23, 25, 27],
    'right_leg': [24, 26, 28],
}

LANDMARK_SMOOTH = 0.5

SKELETON_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
]
DRAW_JOINTS = {11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28}


class LandmarkSmoother:
    def __init__(self, alpha=LANDMARK_SMOOTH, num_landmarks=33):
        self.alpha = alpha
        self.num = num_landmarks
        self.state = None

    def smooth(self, landmarks, w, h):
        if landmarks is None or len(landmarks) == 0:
            if self.state is not None:
                self.state[:, 3] *= 0.85
            return self._to_points(w, h)

        raw = np.zeros((self.num, 4), dtype=np.float64)
        for i, lm in enumerate(landmarks):
            if i >= self.num:
                break
            raw[i] = [lm.x, lm.y, lm.z, lm.visibility]

        if self.state is None:
            self.state = raw.copy()
        else:
            self.state[:, :3] = self.alpha * self.state[:, :3] + (1 - self.alpha) * raw[:, :3]
            self.state[:, 3] = raw[:, 3]

        return self._to_points(w, h)

    def _to_points(self, w, h):
        if self.state is None:
            return {}
        points = {}
        for i in range(self.num):
            if self.state[i, 3] > 0.4:
                points[i] = (int(self.state[i, 0] * w), int(self.state[i, 1] * h))
        return points


def compute_region_movement(prev_points, curr_points):
    result = {}
    for region, indices in REGION_LANDMARKS.items():
        total = 0.0
        count = 0
        for i in indices:
            if i in curr_points and i in prev_points:
                dx = curr_points[i][0] - prev_points[i][0]
                dy = curr_points[i][1] - prev_points[i][1]
                total += np.sqrt(dx*dx + dy*dy)
                count += 1
        result[region] = total / max(count, 1)
    return result


def smooth_waveform(values, kernel_size=11):
    if len(values) < kernel_size:
        return values
    kernel = cv2.getGaussianKernel(kernel_size, 0).flatten()
    kernel /= kernel.sum()
    arr = np.array(values, dtype=np.float64)
    padded = np.pad(arr, kernel_size // 2, mode='edge')
    return np.convolve(padded, kernel, mode='valid').tolist()


def draw_skeleton(frame, points):
    if not points:
        return
    for (a, b) in SKELETON_CONNECTIONS:
        if a in points and b in points:
            cv2.line(frame, points[a], points[b], ACCENT_DIM, 7, cv2.LINE_AA)
    for (a, b) in SKELETON_CONNECTIONS:
        if a in points and b in points:
            cv2.line(frame, points[a], points[b], ACCENT, 2, cv2.LINE_AA)
    for idx in DRAW_JOINTS:
        if idx in points:
            pt = points[idx]
            cv2.circle(frame, pt, 7, ACCENT_DIM, -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 4, ACCENT, -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 2, WHITE_CORE, -1, cv2.LINE_AA)


def draw_overlapping_waveform(frame, region_histories, w, h, wave_height=150):
    """Draw independent overlapping waveforms — each region from baseline, freely crossing."""
    n_samples = len(region_histories[REGION_LABELS[0]])
    if n_samples < 2:
        return

    y_base = h - 20
    y_top = y_base - wave_height

    # Dark background panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y_top - 30), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Smooth each region
    smoothed = {}
    for region in REGION_LABELS:
        smoothed[region] = smooth_waveform(list(region_histories[region]), kernel_size=11)

    # Global max across all regions for consistent scaling
    global_max = 0.0
    for region in REGION_LABELS:
        m = max(smoothed[region]) if smoothed[region] else 0
        if m > global_max:
            global_max = m
    if global_max <= 0:
        global_max = 1.0

    n = len(smoothed[REGION_LABELS[0]])
    step = w / max(n - 1, 1)

    # Sort regions by their current (last) value so the least active draws first (back)
    # and the most active draws last (front) — dynamic z-ordering
    current_vals = {r: smoothed[r][-1] if smoothed[r] else 0 for r in REGION_LABELS}
    draw_order = sorted(REGION_LABELS, key=lambda r: current_vals[r])

    # Draw each region as a clean line (no fill) with glow for visibility
    for region in draw_order:
        color = REGION_COLORS[region]
        vals = smoothed[region]

        pts_top = []
        for i, v in enumerate(vals):
            x = int(i * step)
            y = int(y_base - (v / global_max) * wave_height)
            y = max(y, y_top)  # clamp
            pts_top.append([x, y])

        if len(pts_top) < 2:
            continue

        line_pts = np.array(pts_top, dtype=np.int32).reshape((-1, 1, 2))

        # Glow: dim wider line behind for separation
        dim = tuple(max(c // 3, 0) for c in color)
        cv2.polylines(frame, [line_pts], False, dim, 5, cv2.LINE_AA)

        # Main line — thicker for most active, always bright
        bright = tuple(min(c + 60, 255) for c in color)
        thickness = 3 if region == draw_order[-1] else 2
        cv2.polylines(frame, [line_pts], False, bright, thickness, cv2.LINE_AA)

    # Playhead
    curr_x = int((n - 1) * step)
    cv2.line(frame, (curr_x, y_top - 30), (curr_x, y_base), WHITE_CORE, 1, cv2.LINE_AA)

    # Thin baseline
    cv2.line(frame, (0, y_base), (w, y_base), (80, 80, 80), 1, cv2.LINE_AA)

    # Legend
    legend_y = y_top - 12
    legend_x = 10
    for region in REGION_LABELS:
        color = REGION_COLORS[region]
        label = REGION_DISPLAY[region]
        bright = tuple(min(c + 50, 255) for c in color)
        cv2.rectangle(frame, (legend_x, legend_y - 8), (legend_x + 10, legend_y + 2), bright, -1)
        cv2.putText(frame, label, (legend_x + 14, legend_y + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, WHITE_CORE, 1, cv2.LINE_AA)
        legend_x += 14 + len(label) * 7 + 12


def process(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {input_path}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing: {input_path} — {w}x{h} @ {fps:.1f}fps, {total_frames} frames")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path='/tmp/flekks-viz/pose_landmarker_heavy.task',
        ),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.6,
    )
    landmarker = PoseLandmarker.create_from_options(options)

    smoother = LandmarkSmoother(alpha=LANDMARK_SMOOTH)
    prev_points = {}
    max_history = int(fps * 10)
    region_histories = {r: deque(maxlen=max_history) for r in REGION_LABELS}
    frame_idx = 0
    timestamp_ms = 0
    frame_duration_ms = 1000.0 / fps

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, int(timestamp_ms))
        timestamp_ms += frame_duration_ms

        landmarks = None
        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            landmarks = result.pose_landmarks[0]

        curr_points = smoother.smooth(landmarks, w, h)
        region_move = compute_region_movement(prev_points, curr_points)
        for region in REGION_LABELS:
            region_histories[region].append(region_move[region])
        prev_points = curr_points

        draw_skeleton(frame, curr_points)
        draw_overlapping_waveform(frame, region_histories, w, h)

        out.write(frame)
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  {frame_idx}/{total_frames} ({100*frame_idx//total_frames}%)")

    cap.release()
    out.release()
    landmarker.close()
    print(f"Done: {output_path} ({frame_idx} frames)")


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        process(sys.argv[1], sys.argv[2])
    else:
        process("/tmp/flekks-viz/input.mp4", "/tmp/flekks-viz/output.mp4")
