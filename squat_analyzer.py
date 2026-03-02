#!/usr/bin/env python3
"""
Flekks Squat Asymmetry Analyzer v2 — Clinical Goniometric Standards
- Knee Valgus (FPPA): ASIS→patella→ankle midpoint angle, normal <8° (males), <13° (females)
- Ankle Pronation: heel-to-ankle lateral shift during squat
- Hip Shift: lateral displacement of hip midpoint from shoulder midpoint
- Shoulder Tilt: left vs right shoulder height difference
- Depth Symmetry: left vs right hip depth comparison
- Squat Depth tracking with rep detection
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

# Flekks brand colors (BGR)
ACCENT     = (166, 180, 64)
ACCENT_DIM = (115, 125, 45)
WHITE      = (240, 245, 240)
RED        = (70, 70, 230)
GREEN      = (100, 200, 100)
YELLOW     = (60, 200, 230)
PANEL_BG   = (20, 20, 20)
GRAY       = (100, 100, 100)
LEFT_CLR   = (204, 104, 132)   # pink-ish for left side
RIGHT_CLR  = (58, 112, 196)    # blue-ish for right side

# Landmark indices
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_FOOT, R_FOOT = 31, 32

SKELETON = [
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 29), (28, 30), (27, 31), (28, 32),
]
JOINTS = {11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}

LANDMARK_SMOOTH = 0.45

# Clinical norms for FPPA (Frontal Plane Projection Angle)
FPPA_NORMAL_MALE = 8.0      # degrees
FPPA_NORMAL_FEMALE = 13.0   # degrees (we'll use this as general threshold)
FPPA_WARN = 10.0            # mild valgus threshold
FPPA_BAD = 16.0             # significant valgus (PFP risk)


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
            if self.state[i, 3] > 0.35:
                points[i] = (int(self.state[i, 0] * w), int(self.state[i, 1] * h))
        return points


def midpoint(p1, p2):
    return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)


def compute_fppa(hip, knee, ankle):
    """Compute Frontal Plane Projection Angle (FPPA).
    FPPA = angle at knee formed by hip-knee and ankle-knee vectors in frontal plane.
    Positive = valgus (knee medial), Negative = varus."""
    v_thigh = np.array([hip[0] - knee[0], hip[1] - knee[1]], dtype=np.float64)
    v_shank = np.array([ankle[0] - knee[0], ankle[1] - knee[1]], dtype=np.float64)

    # Full angle between the two vectors
    cos_a = np.dot(v_thigh, v_shank) / (np.linalg.norm(v_thigh) * np.linalg.norm(v_shank) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))

    # Deviation from 180° (straight line)
    deviation = 180.0 - angle

    # Determine direction: cross product sign tells us valgus vs varus
    cross = v_thigh[0] * v_shank[1] - v_thigh[1] * v_shank[0]

    return deviation, cross


def compute_squat_metrics(points):
    """Compute all squat metrics using clinical standards."""
    metrics = {}

    # ── KNEE VALGUS (FPPA) — Left ──
    if L_HIP in points and L_KNEE in points and L_ANKLE in points:
        dev, cross = compute_fppa(points[L_HIP], points[L_KNEE], points[L_ANKLE])
        metrics['l_fppa'] = dev
        metrics['l_fppa_dir'] = 'valgus' if cross > 0 else 'varus'
    else:
        metrics['l_fppa'] = 0.0
        metrics['l_fppa_dir'] = 'neutral'

    # ── KNEE VALGUS (FPPA) — Right ──
    if R_HIP in points and R_KNEE in points and R_ANKLE in points:
        dev, cross = compute_fppa(points[R_HIP], points[R_KNEE], points[R_ANKLE])
        metrics['r_fppa'] = dev
        metrics['r_fppa_dir'] = 'valgus' if cross < 0 else 'varus'
    else:
        metrics['r_fppa'] = 0.0
        metrics['r_fppa_dir'] = 'neutral'

    # ── ANKLE PRONATION ──
    # Compare heel position to ankle — lateral shift indicates pronation
    for side, ankle_i, heel_i, foot_i, prefix in [
        ('left', L_ANKLE, L_HEEL, L_FOOT, 'l'),
        ('right', R_ANKLE, R_HEEL, R_FOOT, 'r'),
    ]:
        if ankle_i in points and heel_i in points:
            ankle_pt = points[ankle_i]
            heel_pt = points[heel_i]
            # Pronation: ankle rolls inward = ankle x shifts toward midline
            dx = ankle_pt[0] - heel_pt[0]
            leg_len = 1.0
            if (L_HIP if side == 'left' else R_HIP) in points:
                hip_pt = points[L_HIP if side == 'left' else R_HIP]
                leg_len = max(np.sqrt((ankle_pt[0]-hip_pt[0])**2 + (ankle_pt[1]-hip_pt[1])**2), 1)
            metrics[f'{prefix}_pronation'] = (dx / leg_len) * 100
        else:
            metrics[f'{prefix}_pronation'] = 0.0

    # ── HIP SHIFT ──
    mid_shoulders = None
    mid_hips = None
    if L_SHOULDER in points and R_SHOULDER in points:
        mid_shoulders = midpoint(points[L_SHOULDER], points[R_SHOULDER])
    if L_HIP in points and R_HIP in points:
        mid_hips = midpoint(points[L_HIP], points[R_HIP])

    if mid_shoulders and mid_hips:
        shoulder_width = abs(points[R_SHOULDER][0] - points[L_SHOULDER][0])
        shift = (mid_hips[0] - mid_shoulders[0]) / max(shoulder_width, 1) * 100
        metrics['hip_shift'] = shift
    else:
        metrics['hip_shift'] = 0.0

    # ── SHOULDER TILT ──
    if L_SHOULDER in points and R_SHOULDER in points:
        dy = points[R_SHOULDER][1] - points[L_SHOULDER][1]
        dx = abs(points[R_SHOULDER][0] - points[L_SHOULDER][0])
        metrics['shoulder_tilt'] = np.degrees(np.arctan2(abs(dy), max(dx, 1)))
        metrics['shoulder_tilt_dir'] = 'L high' if dy > 0 else 'R high'
    else:
        metrics['shoulder_tilt'] = 0.0
        metrics['shoulder_tilt_dir'] = 'level'

    # ── DEPTH ASYMMETRY ──
    if L_HIP in points and R_HIP in points:
        diff = abs(points[L_HIP][1] - points[R_HIP][1])
        ref = abs(points[L_SHOULDER][1] - points[L_HIP][1]) if L_SHOULDER in points else 100
        metrics['depth_asymmetry'] = (diff / max(ref, 1)) * 100
    else:
        metrics['depth_asymmetry'] = 0.0

    # ── SQUAT DEPTH ──
    if L_HIP in points and R_HIP in points and L_KNEE in points and R_KNEE in points:
        hip_y = (points[L_HIP][1] + points[R_HIP][1]) / 2
        knee_y = (points[L_KNEE][1] + points[R_KNEE][1]) / 2
        # Depth ratio: hip below knee = deep squat
        if L_SHOULDER in points and L_ANKLE in points:
            total_h = points[L_ANKLE][1] - points[L_SHOULDER][1]
            hip_drop = hip_y - points[L_SHOULDER][1]
            metrics['squat_depth'] = (hip_drop / max(total_h, 1)) * 100
        else:
            metrics['squat_depth'] = 50.0
        metrics['hip_below_knee'] = hip_y > knee_y
    else:
        metrics['squat_depth'] = 50.0
        metrics['hip_below_knee'] = False

    metrics['mid_shoulders'] = mid_shoulders
    metrics['mid_hips'] = mid_hips

    return metrics


def fppa_color(fppa):
    if fppa < FPPA_WARN:
        return GREEN
    elif fppa < FPPA_BAD:
        return YELLOW
    else:
        return RED


def draw_skeleton(frame, points):
    if not points:
        return
    for (a, b) in SKELETON:
        if a in points and b in points:
            cv2.line(frame, points[a], points[b], ACCENT_DIM, 5, cv2.LINE_AA)
    for (a, b) in SKELETON:
        if a in points and b in points:
            cv2.line(frame, points[a], points[b], ACCENT, 2, cv2.LINE_AA)
    for idx in JOINTS:
        if idx in points:
            pt = points[idx]
            cv2.circle(frame, pt, 5, ACCENT_DIM, -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 3, ACCENT, -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 2, WHITE, -1, cv2.LINE_AA)


def draw_valgus_overlay(frame, points, metrics):
    """Draw FPPA knee valgus indicators and ideal alignment lines."""

    for hip_i, knee_i, ankle_i, fppa_key, dir_key, side_clr, label in [
        (L_HIP, L_KNEE, L_ANKLE, 'l_fppa', 'l_fppa_dir', LEFT_CLR, 'L'),
        (R_HIP, R_KNEE, R_ANKLE, 'r_fppa', 'r_fppa_dir', RIGHT_CLR, 'R'),
    ]:
        if hip_i not in points or knee_i not in points or ankle_i not in points:
            continue

        hip_pt = points[hip_i]
        knee_pt = points[knee_i]
        ankle_pt = points[ankle_i]

        # Draw ideal straight line (hip → ankle)
        cv2.line(frame, hip_pt, ankle_pt, GRAY, 1, cv2.LINE_AA)

        # Compute where knee SHOULD be on that line
        t = (knee_pt[1] - hip_pt[1]) / max(ankle_pt[1] - hip_pt[1], 1)
        ideal_x = int(hip_pt[0] + t * (ankle_pt[0] - hip_pt[0]))
        ideal_knee = (ideal_x, knee_pt[1])

        # Arrow from ideal to actual knee position
        fppa = metrics[fppa_key]
        color = fppa_color(fppa)

        if abs(knee_pt[0] - ideal_x) > 3:
            cv2.arrowedLine(frame, ideal_knee, knee_pt, color, 2, cv2.LINE_AA, tipLength=0.25)

        # FPPA angle text near knee
        direction = metrics[dir_key]
        text = f'{label} {fppa:.1f}° {direction}'
        text_x = knee_pt[0] + (15 if label == 'R' else -120)
        cv2.putText(frame, text, (text_x, knee_pt[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)

    # ── HIP SHIFT LINE ──
    mid_s = metrics.get('mid_shoulders')
    mid_h = metrics.get('mid_hips')
    if mid_s and mid_h:
        # Vertical plumb from shoulder midpoint
        cv2.line(frame, (mid_s[0], mid_s[1] - 20), (mid_s[0], mid_s[1] + 250),
                 GRAY, 1, cv2.LINE_AA)
        # Hip offset arrow
        if abs(mid_h[0] - mid_s[0]) > 3:
            shift_color = GREEN if abs(metrics['hip_shift']) < 3 else (YELLOW if abs(metrics['hip_shift']) < 8 else RED)
            cv2.arrowedLine(frame, (mid_s[0], mid_h[1]), mid_h,
                            shift_color, 2, cv2.LINE_AA, tipLength=0.3)


def draw_pronation_indicators(frame, points, metrics):
    """Draw ankle pronation indicators."""
    for ankle_i, heel_i, prefix, label in [
        (L_ANKLE, L_HEEL, 'l', 'L'),
        (R_ANKLE, R_HEEL, 'r', 'R'),
    ]:
        if ankle_i in points and heel_i in points:
            ankle_pt = points[ankle_i]
            heel_pt = points[heel_i]
            pron = metrics[f'{prefix}_pronation']

            color = GREEN if abs(pron) < 2 else (YELLOW if abs(pron) < 5 else RED)

            # Small indicator line from heel to ankle
            cv2.line(frame, heel_pt, ankle_pt, color, 2, cv2.LINE_AA)

            text = f'{label} Pron {pron:.1f}%'
            cv2.putText(frame, text, (ankle_pt[0] - 30, ankle_pt[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1, cv2.LINE_AA)


def draw_metrics_panel(frame, metrics, w, h):
    """Draw clinical metrics panel."""
    panel_w = 250
    panel_x = w - panel_w - 10
    panel_y = 10
    row_h = 24

    items = [
        ('L KNEE VALGUS (FPPA)', f'{metrics["l_fppa"]:.1f}° {metrics["l_fppa_dir"]}',
         fppa_color(metrics['l_fppa']), f'Normal: <{FPPA_WARN:.0f}°'),
        ('R KNEE VALGUS (FPPA)', f'{metrics["r_fppa"]:.1f}° {metrics["r_fppa_dir"]}',
         fppa_color(metrics['r_fppa']), f'Normal: <{FPPA_WARN:.0f}°'),
        ('L ANKLE PRONATION', f'{metrics["l_pronation"]:.1f}%',
         GREEN if abs(metrics['l_pronation']) < 2 else (YELLOW if abs(metrics['l_pronation']) < 5 else RED),
         'Normal: <2%'),
        ('R ANKLE PRONATION', f'{metrics["r_pronation"]:.1f}%',
         GREEN if abs(metrics['r_pronation']) < 2 else (YELLOW if abs(metrics['r_pronation']) < 5 else RED),
         'Normal: <2%'),
        ('HIP SHIFT', f'{metrics["hip_shift"]:+.1f}%',
         GREEN if abs(metrics['hip_shift']) < 3 else (YELLOW if abs(metrics['hip_shift']) < 8 else RED),
         'Normal: <3%'),
        ('SHOULDER TILT', f'{metrics["shoulder_tilt"]:.1f}° {metrics["shoulder_tilt_dir"]}',
         GREEN if metrics['shoulder_tilt'] < 2 else (YELLOW if metrics['shoulder_tilt'] < 5 else RED),
         'Normal: <2°'),
        ('DEPTH ASYM', f'{metrics["depth_asymmetry"]:.1f}%',
         GREEN if metrics['depth_asymmetry'] < 3 else (YELLOW if metrics['depth_asymmetry'] < 8 else RED),
         'Normal: <3%'),
    ]

    panel_h = len(items) * (row_h + 10) + 70

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x - 5, panel_y - 5),
                  (panel_x + panel_w + 5, panel_y + panel_h), PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.putText(frame, 'SQUAT ANALYSIS', (panel_x + 10, panel_y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, ACCENT, 1, cv2.LINE_AA)

    # Squat depth indicator
    depth = metrics['squat_depth']
    depth_text = f'Depth: {depth:.0f}%'
    if metrics['hip_below_knee']:
        depth_text += ' (PARALLEL+)'
    cv2.putText(frame, depth_text, (panel_x + 10, panel_y + 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, YELLOW if depth > 60 else WHITE, 1, cv2.LINE_AA)

    y = panel_y + 56
    for label, value_str, color, norm_str in items:
        cv2.circle(frame, (panel_x + 10, y - 3), 3, color, -1, cv2.LINE_AA)
        cv2.putText(frame, label, (panel_x + 18, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.27, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, value_str, (panel_x + 170, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)
        cv2.putText(frame, norm_str, (panel_x + 18, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.22, GRAY, 1, cv2.LINE_AA)
        y += row_h + 10

    # Findings
    y += 5
    findings = []
    if metrics['l_fppa'] > FPPA_BAD:
        findings.append('L knee: significant valgus')
    elif metrics['l_fppa'] > FPPA_WARN:
        findings.append('L knee: mild valgus')
    if metrics['r_fppa'] > FPPA_BAD:
        findings.append('R knee: significant valgus')
    elif metrics['r_fppa'] > FPPA_WARN:
        findings.append('R knee: mild valgus')
    if abs(metrics['l_pronation']) > 5 or abs(metrics['r_pronation']) > 5:
        findings.append('Excessive ankle pronation')
    if abs(metrics['hip_shift']) > 8:
        findings.append('Significant lateral hip shift')

    for f in findings[:3]:
        cv2.putText(frame, f'• {f}', (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.27, RED, 1, cv2.LINE_AA)
        y += 15

    if not findings:
        cv2.putText(frame, '• Good form', (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.27, GREEN, 1, cv2.LINE_AA)


def draw_depth_graph(frame, depth_history, w, h):
    """Small squat depth graph at bottom-left."""
    if len(depth_history) < 2:
        return

    graph_w, graph_h = 200, 60
    gx, gy = 10, h - graph_h - 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (gx, gy), (gx + graph_w, gy + graph_h), PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, 'SQUAT DEPTH', (gx + 5, gy + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, GRAY, 1, cv2.LINE_AA)

    vals = list(depth_history)
    n = len(vals)
    step = graph_w / max(n - 1, 1)

    pts = []
    for i, v in enumerate(vals):
        x = int(gx + i * step)
        y_val = int(gy + graph_h - (v / 100.0) * graph_h)
        y_val = max(gy, min(gy + graph_h, y_val))
        pts.append([x, y_val])

    if len(pts) >= 2:
        line_pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [line_pts], False, ACCENT, 2, cv2.LINE_AA)


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
    depth_history = deque(maxlen=int(fps * 10))
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
        metrics = compute_squat_metrics(curr_points)
        depth_history.append(metrics['squat_depth'])

        draw_skeleton(frame, curr_points)
        draw_valgus_overlay(frame, curr_points, metrics)
        draw_pronation_indicators(frame, curr_points, metrics)
        draw_metrics_panel(frame, metrics, w, h)
        draw_depth_graph(frame, depth_history, w, h)

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
        process("/tmp/flekks-viz/input.mp4", "/tmp/flekks-viz/squat_output.mp4")
