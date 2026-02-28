#!/usr/bin/env python3
"""
Flekks Posture Analyzer v2 (Side Profile) — Clinical Goniometric Standards
- Craniovertebral Angle (CVA): tragus → C7, normal ≥50°, FHP <48°
- Thoracic Kyphosis: shoulder→mid-spine→hip curvature, normal 20-40°
- Lumbar Lordosis: mid-spine→hip→knee alignment, normal 30-50°
- Anterior Pelvic Tilt: ASIS-PSIS line vs horizontal, normal 7-15°
- Forward Rounded Shoulders: shoulder offset anterior to plumb line
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
IDEAL_CLR  = (80, 140, 60)
PANEL_BG   = (20, 20, 20)
GRAY       = (100, 100, 100)

# Landmark indices
EAR_L, EAR_R = 7, 8
SHOULDER_L, SHOULDER_R = 11, 12
HIP_L, HIP_R = 23, 24
KNEE_L, KNEE_R = 25, 26
ANKLE_L, ANKLE_R = 27, 28

SKELETON = [
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (23, 25), (25, 27), (24, 26), (26, 28),
]
JOINTS = {7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28}

LANDMARK_SMOOTH = 0.45

# Clinical norms
CVA_NORMAL = 50.0        # degrees, >= is normal
CVA_FHP = 48.0           # degrees, < is forward head posture
KYPHOSIS_MIN = 20.0      # normal thoracic kyphosis range
KYPHOSIS_MAX = 40.0
LORDOSIS_MIN = 30.0      # normal lumbar lordosis range
LORDOSIS_MAX = 50.0
PELVIC_TILT_MIN = 7.0    # normal anterior pelvic tilt range
PELVIC_TILT_MAX = 15.0
SHOULDER_PROTRACTION_THRESH = 5.0  # degrees forward of plumb


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
            if self.state[i, 3] > 0.3:
                points[i] = (int(self.state[i, 0] * w), int(self.state[i, 1] * h))
        return points


def pick_side(points):
    """Pick the more visible side for side-profile analysis."""
    left_vis = sum(1 for i in [EAR_L, SHOULDER_L, HIP_L, KNEE_L, ANKLE_L] if i in points)
    right_vis = sum(1 for i in [EAR_R, SHOULDER_R, HIP_R, KNEE_R, ANKLE_R] if i in points)
    if left_vis >= right_vis:
        return EAR_L, SHOULDER_L, HIP_L, KNEE_L, ANKLE_L
    else:
        return EAR_R, SHOULDER_R, HIP_R, KNEE_R, ANKLE_R


def angle_at(p1, vertex, p2):
    """Angle at vertex formed by p1-vertex-p2 in degrees."""
    v1 = np.array([p1[0] - vertex[0], p1[1] - vertex[1]], dtype=np.float64)
    v2 = np.array([p2[0] - vertex[0], p2[1] - vertex[1]], dtype=np.float64)
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))


def compute_posture(points):
    """Compute posture metrics using clinical goniometric standards."""
    ear_i, shoulder_i, hip_i, knee_i, ankle_i = pick_side(points)
    metrics = {}
    chain = {}

    for label, idx in [('ear', ear_i), ('shoulder', shoulder_i),
                       ('hip', hip_i), ('knee', knee_i), ('ankle', ankle_i)]:
        if idx in points:
            chain[label] = points[idx]

    metrics['chain'] = chain

    # ── CRANIOVERTEBRAL ANGLE (CVA) ──
    # Angle between horizontal line through C7 (≈shoulder) and line from C7 to tragus (ear)
    # Larger = more upright. Normal ≥50°, FHP <48°
    if 'ear' in chain and 'shoulder' in chain:
        ear, shoulder = chain['ear'], chain['shoulder']
        # Vector from shoulder to ear
        dx = ear[0] - shoulder[0]
        dy = shoulder[1] - ear[1]  # flip because y increases downward
        # Angle from horizontal
        cva = np.degrees(np.arctan2(dy, abs(dx)))
        metrics['cva'] = cva
    else:
        metrics['cva'] = 50.0

    # ── FORWARD SHOULDER ──
    # How far the shoulder is anterior (forward) of the plumb line from ankle
    # Measured as angle from vertical
    if 'shoulder' in chain and 'hip' in chain:
        shoulder, hip = chain['shoulder'], chain['hip']
        dx = shoulder[0] - hip[0]  # positive = shoulder forward of hip
        dy = hip[1] - shoulder[1]
        metrics['shoulder_protraction'] = np.degrees(np.arctan2(abs(dx), max(dy, 1)))
    else:
        metrics['shoulder_protraction'] = 0.0

    # ── THORACIC KYPHOSIS ESTIMATE ──
    # Angle at the mid-thoracic region: ear-shoulder-hip
    # In ideal posture this is close to 180° (straight line)
    # Kyphosis deviation = 180 - this angle
    # Normal kyphosis: 20-40°
    if 'ear' in chain and 'shoulder' in chain and 'hip' in chain:
        upper_angle = angle_at(chain['ear'], chain['shoulder'], chain['hip'])
        kyphosis_est = 180.0 - upper_angle
        metrics['thoracic_kyphosis'] = max(0, kyphosis_est)
    else:
        metrics['thoracic_kyphosis'] = 30.0

    # ── LUMBAR LORDOSIS ESTIMATE ──
    # Angle at hip: shoulder-hip-knee
    # Deviation from 180° indicates lordosis
    # Normal: 30-50°
    if 'shoulder' in chain and 'hip' in chain and 'knee' in chain:
        hip_angle = angle_at(chain['shoulder'], chain['hip'], chain['knee'])
        lordosis_est = 180.0 - hip_angle
        metrics['lumbar_lordosis'] = max(0, lordosis_est)
    else:
        metrics['lumbar_lordosis'] = 40.0

    # ── ANTERIOR PELVIC TILT ──
    # Angle of the hip-to-knee line from vertical
    # Normal: 7-15°, >15° = excessive anterior tilt
    if 'hip' in chain and 'knee' in chain:
        hip, knee = chain['hip'], chain['knee']
        dx = knee[0] - hip[0]
        dy = knee[1] - hip[1]
        tilt = np.degrees(np.arctan2(abs(dx), max(abs(dy), 1)))
        metrics['pelvic_tilt'] = tilt
    else:
        metrics['pelvic_tilt'] = 10.0

    # ── KNEE HYPEREXTENSION ──
    # Angle at knee: hip-knee-ankle
    # Normal ≈ 180° (straight), <175° = flexed, >185° = hyperextended
    if 'hip' in chain and 'knee' in chain and 'ankle' in chain:
        metrics['knee_angle'] = angle_at(chain['hip'], chain['knee'], chain['ankle'])
    else:
        metrics['knee_angle'] = 180.0

    # ── OVERALL ALIGNMENT SCORE ──
    issues = 0
    # CVA
    if metrics['cva'] < CVA_FHP:
        issues += 2
    elif metrics['cva'] < CVA_NORMAL:
        issues += 1
    # Kyphosis
    if metrics['thoracic_kyphosis'] > KYPHOSIS_MAX:
        issues += 2
    elif metrics['thoracic_kyphosis'] < KYPHOSIS_MIN:
        issues += 1
    # Lordosis
    if metrics['lumbar_lordosis'] > LORDOSIS_MAX:
        issues += 2
    elif metrics['lumbar_lordosis'] < LORDOSIS_MIN:
        issues += 1
    # Pelvic tilt
    if metrics['pelvic_tilt'] > PELVIC_TILT_MAX:
        issues += 2
    elif metrics['pelvic_tilt'] < PELVIC_TILT_MIN:
        issues += 1
    # Shoulder protraction
    if metrics['shoulder_protraction'] > SHOULDER_PROTRACTION_THRESH:
        issues += 1

    metrics['score'] = max(0, 100 - issues * 12)

    return metrics


def range_color(value, low, high):
    """Green if within range, yellow if close, red if far outside."""
    if low <= value <= high:
        return GREEN
    elif (low - 5) <= value <= (high + 5):
        return YELLOW
    else:
        return RED


def cva_color(cva):
    if cva >= CVA_NORMAL:
        return GREEN
    elif cva >= CVA_FHP:
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
            cv2.circle(frame, pt, 6, ACCENT_DIM, -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 3, ACCENT, -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 2, WHITE, -1, cv2.LINE_AA)


def draw_posture_overlay(frame, metrics, w, h):
    """Draw clinical posture analysis overlay."""
    chain = metrics.get('chain', {})
    if len(chain) < 3:
        return

    # ── PLUMB LINE from ankle ──
    if 'ankle' in chain:
        ankle = chain['ankle']
        top_y = min(chain.get('ear', chain.get('shoulder', (0, 50)))[1] - 40, 20)
        cv2.line(frame, (ankle[0], top_y), (ankle[0], ankle[1]),
                 IDEAL_CLR, 1, cv2.LINE_AA)
        cv2.putText(frame, 'PLUMB', (ankle[0] - 22, top_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, IDEAL_CLR, 1, cv2.LINE_AA)

    # ── POSTURE CHAIN (actual) ──
    chain_order = ['ear', 'shoulder', 'hip', 'knee', 'ankle']
    prev = None
    for key in chain_order:
        if key in chain:
            pt = chain[key]
            if prev is not None:
                cv2.line(frame, prev, pt, ACCENT, 3, cv2.LINE_AA)
            cv2.circle(frame, pt, 8, ACCENT, -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 5, WHITE, -1, cv2.LINE_AA)
            # Label on the opposite side of plumb line
            offset = 15 if 'ankle' not in chain or pt[0] >= chain.get('ankle', pt)[0] else -60
            cv2.putText(frame, key.upper(), (pt[0] + offset, pt[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, WHITE, 1, cv2.LINE_AA)
            prev = pt

    # ── CVA ARC at shoulder ──
    if 'ear' in chain and 'shoulder' in chain:
        ear, shoulder = chain['ear'], chain['shoulder']
        color = cva_color(metrics['cva'])
        # Draw the CVA angle arc
        horiz_pt = (shoulder[0] + 50, shoulder[1])
        _draw_arc(frame, shoulder, horiz_pt, ear, color, f"CVA {metrics['cva']:.0f}°")

    # ── KYPHOSIS indicator at shoulder ──
    if 'ear' in chain and 'shoulder' in chain and 'hip' in chain:
        color = range_color(metrics['thoracic_kyphosis'], KYPHOSIS_MIN, KYPHOSIS_MAX)
        # Small angle indicator
        shoulder = chain['shoulder']
        label = f"T-Kyph {metrics['thoracic_kyphosis']:.0f}°"
        if metrics['thoracic_kyphosis'] > KYPHOSIS_MAX:
            label += " EXCESS"
        elif metrics['thoracic_kyphosis'] < KYPHOSIS_MIN:
            label += " FLAT"
        cv2.putText(frame, label, (shoulder[0] - 80, shoulder[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)

    # ── LORDOSIS indicator at hip ──
    if 'shoulder' in chain and 'hip' in chain and 'knee' in chain:
        color = range_color(metrics['lumbar_lordosis'], LORDOSIS_MIN, LORDOSIS_MAX)
        hip = chain['hip']
        label = f"L-Lord {metrics['lumbar_lordosis']:.0f}°"
        if metrics['lumbar_lordosis'] > LORDOSIS_MAX:
            label += " EXCESS"
        elif metrics['lumbar_lordosis'] < LORDOSIS_MIN:
            label += " FLAT"
        cv2.putText(frame, label, (hip[0] - 80, hip[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)

    # ── PELVIC TILT indicator ──
    if 'hip' in chain and 'knee' in chain:
        color = range_color(metrics['pelvic_tilt'], PELVIC_TILT_MIN, PELVIC_TILT_MAX)
        hip = chain['hip']
        label = f"Pelv Tilt {metrics['pelvic_tilt']:.0f}°"
        if metrics['pelvic_tilt'] > PELVIC_TILT_MAX:
            label += " ANT"
        cv2.putText(frame, label, (hip[0] - 80, hip[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)


def _draw_arc(frame, vertex, p1, p2, color, label):
    """Draw a small angle arc with label."""
    radius = 30
    a1 = np.degrees(np.arctan2(p1[1] - vertex[1], p1[0] - vertex[0]))
    a2 = np.degrees(np.arctan2(p2[1] - vertex[1], p2[0] - vertex[0]))
    # Ensure we draw the smaller arc
    if abs(a2 - a1) > 180:
        if a2 > a1:
            a1 += 360
        else:
            a2 += 360
    cv2.ellipse(frame, vertex, (radius, radius), 0, min(a1, a2), max(a1, a2),
                color, 2, cv2.LINE_AA)
    mid_a = np.radians((a1 + a2) / 2)
    tx = int(vertex[0] + (radius + 18) * np.cos(mid_a))
    ty = int(vertex[1] + (radius + 18) * np.sin(mid_a))
    cv2.putText(frame, label, (tx - 30, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)


def draw_metrics_panel(frame, metrics, w, h):
    """Draw clinical metrics panel."""
    panel_w = 260
    panel_x = w - panel_w - 10
    panel_y = 10
    row_h = 28

    items = [
        ('HEAD POSITION (CVA)', f'{metrics["cva"]:.1f}°', cva_color(metrics['cva']),
         f'Normal: ≥{CVA_NORMAL:.0f}°'),
        ('SHOULDER PROTRACTION', f'{metrics["shoulder_protraction"]:.1f}°',
         GREEN if metrics['shoulder_protraction'] < 5 else (YELLOW if metrics['shoulder_protraction'] < 10 else RED),
         'Normal: <5°'),
        ('THORACIC KYPHOSIS', f'{metrics["thoracic_kyphosis"]:.0f}°',
         range_color(metrics['thoracic_kyphosis'], KYPHOSIS_MIN, KYPHOSIS_MAX),
         f'Normal: {KYPHOSIS_MIN:.0f}-{KYPHOSIS_MAX:.0f}°'),
        ('LUMBAR LORDOSIS', f'{metrics["lumbar_lordosis"]:.0f}°',
         range_color(metrics['lumbar_lordosis'], LORDOSIS_MIN, LORDOSIS_MAX),
         f'Normal: {LORDOSIS_MIN:.0f}-{LORDOSIS_MAX:.0f}°'),
        ('PELVIC TILT', f'{metrics["pelvic_tilt"]:.1f}°',
         range_color(metrics['pelvic_tilt'], PELVIC_TILT_MIN, PELVIC_TILT_MAX),
         f'Normal: {PELVIC_TILT_MIN:.0f}-{PELVIC_TILT_MAX:.0f}°'),
        ('KNEE ANGLE', f'{metrics["knee_angle"]:.0f}°',
         GREEN if 170 <= metrics['knee_angle'] <= 185 else YELLOW,
         'Normal: ~180°'),
    ]

    panel_h = len(items) * (row_h + 12) + 60

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x - 5, panel_y - 5),
                  (panel_x + panel_w + 5, panel_y + panel_h), PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.putText(frame, 'POSTURE ANALYSIS', (panel_x + 10, panel_y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, ACCENT, 1, cv2.LINE_AA)

    y = panel_y + 42
    for label, value_str, color, norm_str in items:
        cv2.circle(frame, (panel_x + 10, y - 3), 4, color, -1, cv2.LINE_AA)
        cv2.putText(frame, label, (panel_x + 20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, value_str, (panel_x + 185, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
        cv2.putText(frame, norm_str, (panel_x + 20, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, GRAY, 1, cv2.LINE_AA)
        y += row_h + 12

    # Overall
    score = metrics['score']
    if score >= 80:
        grade, gc = 'GOOD POSTURE', GREEN
    elif score >= 55:
        grade, gc = 'FAIR — MINOR DEVIATIONS', YELLOW
    else:
        grade, gc = 'NEEDS ATTENTION', RED

    y += 5
    cv2.putText(frame, f'{grade} ({score}/100)', (panel_x + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, gc, 1, cv2.LINE_AA)

    # Specific findings
    y += 22
    findings = []
    if metrics['cva'] < CVA_FHP:
        findings.append('Forward head posture detected')
    if metrics['thoracic_kyphosis'] > KYPHOSIS_MAX:
        findings.append('Excessive thoracic kyphosis')
    elif metrics['thoracic_kyphosis'] < KYPHOSIS_MIN:
        findings.append('Decreased thoracic kyphosis (flat)')
    if metrics['lumbar_lordosis'] > LORDOSIS_MAX:
        findings.append('Excessive lumbar lordosis')
    elif metrics['lumbar_lordosis'] < LORDOSIS_MIN:
        findings.append('Decreased lordosis (flat low back)')
    if metrics['pelvic_tilt'] > PELVIC_TILT_MAX:
        findings.append('Excessive anterior pelvic tilt')
    if metrics['shoulder_protraction'] > SHOULDER_PROTRACTION_THRESH:
        findings.append('Forward rounded shoulders')

    for finding in findings[:4]:
        cv2.putText(frame, f'• {finding}', (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, RED, 1, cv2.LINE_AA)
        y += 16

    if not findings:
        cv2.putText(frame, '• No major deviations', (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, GREEN, 1, cv2.LINE_AA)


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
        metrics = compute_posture(curr_points)

        draw_skeleton(frame, curr_points)
        draw_posture_overlay(frame, metrics, w, h)
        draw_metrics_panel(frame, metrics, w, h)

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
        process("/tmp/flekks-viz/input.mp4", "/tmp/flekks-viz/posture_output.mp4")
