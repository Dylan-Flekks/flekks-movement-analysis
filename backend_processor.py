#!/usr/bin/env python3
"""
Flekks Backend Processor — Dual Output (Rendered Video + JSON)

Single entry point for server-side video processing. Produces:
  - Rendered video (.mp4) with skeleton + arcs + panels + waveforms  (--output-video)
  - JSON to stdout: MovementFrameTrack + bodyChannels + analysis summary

Usage:
    python3 backend_processor.py <input.mp4> [--output-video <path>] [--channels]

Dependencies:
    pip3 install mediapipe opencv-python numpy
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import json
import sys
import os
import math

# ── Re-use from gait_analyzer ──
from gait_analyzer import (
    # Smoothers + trackers
    Smoother, WorldSmoother, ROMTracker, GaitTracker,
    # Geometry
    angle_at_3d, trunk_axis, angle_between_vectors, normalize_vec,
    angle_from_vertical_3d, compute_ankle_df_pf_3d,
    gonio_shoulder_flex, gonio_hip_abd, gonio_shoulder_abd,
    mid, angle_at, detect_view, compute_movement, smooth_wave, norm_color,
    # Angle computation
    compute_joint_angles,
    # Drawing
    draw_skeleton, draw_arc, draw_joint_angles,
    draw_front_guides, draw_frontal_panel, draw_sagittal_panel, draw_waveform,
    # Constants
    SMOOTH_ALPHA, REGION_LABELS, REGION_LANDMARKS, REGION_COLORS, REGION_DISPLAY,
    SHOULDER_L, SHOULDER_R, ELBOW_L, ELBOW_R, WRIST_L, WRIST_R,
    HIP_L, HIP_R, KNEE_L, KNEE_R, ANKLE_L, ANKLE_R,
    HEEL_L, HEEL_R, FOOT_L, FOOT_R, NOSE, EAR_L, EAR_R,
)

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# ── Model path ──
MODEL_PATH = os.environ.get(
    "POSE_MODEL_PATH",
    "/home/peters/adha/models/pose_landmarker_heavy.task"
)
# Fallback for local Mac development
if not os.path.exists(MODEL_PATH):
    LOCAL_PATH = "/tmp/flekks-viz/pose_landmarker_heavy.task"
    if os.path.exists(LOCAL_PATH):
        MODEL_PATH = LOCAL_PATH

# ── Compression parameters (match iOS FrameDataCollector) ──
KEYFRAME_INTERVAL = 15
ANGLE_DELTA_THRESHOLD = 2.0
POS_DELTA_THRESHOLD_3D = 0.005
POS_DELTA_THRESHOLD_2D = 0.003
VISIBILITY_THRESHOLD = 0.3

# ── ARKit joint name mapping ──
ARKIT_JOINT_MAP = {
    "hips_joint": "midpoint_hips",
    "spine_1_joint": "interp_spine_1",
    "spine_4_joint": "interp_spine_4",
    "spine_7_joint": "interp_spine_7",
    "neck_1_joint": "midpoint_shoulders",
    "head_joint": NOSE,
    "left_shoulder_1_joint": SHOULDER_L,
    "right_shoulder_1_joint": SHOULDER_R,
    "left_arm_joint": ELBOW_L,
    "right_arm_joint": ELBOW_R,
    "left_hand_joint": WRIST_L,
    "right_hand_joint": WRIST_R,
    "left_upLeg_joint": HIP_L,
    "right_upLeg_joint": HIP_R,
    "left_leg_joint": KNEE_L,
    "right_leg_joint": KNEE_R,
    "left_foot_joint": ANKLE_L,
    "right_foot_joint": ANKLE_R,
}


# ── 3D → iOS angle mapping ──

def map_to_ios_angles(a):
    """Map gait_analyzer internal angle dict to iOS JointAngles struct keys."""
    return {
        "leftKnee": a.get('l_knee', 180),
        "rightKnee": a.get('r_knee', 180),
        "leftHip": a.get('l_hip', 180),
        "rightHip": a.get('r_hip', 180),
        "leftElbow": a.get('l_elbow', 180),
        "rightElbow": a.get('r_elbow', 180),
        "leftShoulder": 180 - a.get('l_shoulder', 0),   # gonio → included
        "rightShoulder": 180 - a.get('r_shoulder', 0),
        "spineTilt": a.get('trunk', 0),
        "hipShift": a.get('hip_shift', 0),
        "leftKneeValgus": a.get('l_fppa', 0),
        "rightKneeValgus": a.get('r_fppa', 0),
    }


# ── ARKit position computation ──

def compute_arkit_positions(smoother_state, world_landmarks, w, h):
    """Compute ARKit-named joint positions from smoothed landmarks.
    smoother_state: the Smoother's internal numpy state (33x4)
    world_landmarks: raw MediaPipe world_landmarks[0] list
    Returns (positions_3d, positions_2d)
    """
    positions_3d = {}
    positions_2d = {}

    if smoother_state is None:
        return positions_3d, positions_2d

    state = smoother_state

    def vis(idx):
        return state[idx, 3] > VISIBILITY_THRESHOLD if idx < len(state) else False

    def norm_2d(idx):
        x = float(state[idx, 0])
        y = 1.0 - float(state[idx, 1])  # flip y for Vision compat
        return [round(x, 4), round(y, 4)]

    def world_3d(idx):
        if world_landmarks and idx < len(world_landmarks):
            wl = world_landmarks[idx]
            return [round(wl.x, 5), round(wl.y, 5), round(wl.z, 5)]
        return None

    def midpoint_2d(a, b):
        pa, pb = norm_2d(a), norm_2d(b)
        return [round((pa[0]+pb[0])/2, 4), round((pa[1]+pb[1])/2, 4)]

    def midpoint_3d(a, b):
        wa, wb = world_3d(a), world_3d(b)
        if wa and wb:
            return [round((wa[0]+wb[0])/2, 5), round((wa[1]+wb[1])/2, 5), round((wa[2]+wb[2])/2, 5)]
        return None

    def interp(a, b, t):
        if a and b:
            return [round(a[i]+(b[i]-a[i])*t, 5) for i in range(len(a))]
        return None

    # Hips midpoint
    if vis(HIP_L) and vis(HIP_R):
        positions_2d["hips_joint"] = midpoint_2d(HIP_L, HIP_R)
        w3 = midpoint_3d(HIP_L, HIP_R)
        if w3: positions_3d["hips_joint"] = w3

    # Neck midpoint
    if vis(SHOULDER_L) and vis(SHOULDER_R):
        positions_2d["neck_1_joint"] = midpoint_2d(SHOULDER_L, SHOULDER_R)
        w3 = midpoint_3d(SHOULDER_L, SHOULDER_R)
        if w3: positions_3d["neck_1_joint"] = w3

    # Spine interpolation
    h2d = positions_2d.get("hips_joint")
    n2d = positions_2d.get("neck_1_joint")
    h3d = positions_3d.get("hips_joint")
    n3d = positions_3d.get("neck_1_joint")
    if h2d and n2d:
        for name, t in [("spine_1_joint", 0.25), ("spine_4_joint", 0.50), ("spine_7_joint", 0.75)]:
            positions_2d[name] = [round(h2d[i]+(n2d[i]-h2d[i])*t, 4) for i in range(2)]
    if h3d and n3d:
        for name, t in [("spine_1_joint", 0.25), ("spine_4_joint", 0.50), ("spine_7_joint", 0.75)]:
            positions_3d[name] = interp(h3d, n3d, t)

    # Direct landmarks
    direct = {
        "head_joint": NOSE,
        "left_shoulder_1_joint": SHOULDER_L, "right_shoulder_1_joint": SHOULDER_R,
        "left_arm_joint": ELBOW_L, "right_arm_joint": ELBOW_R,
        "left_hand_joint": WRIST_L, "right_hand_joint": WRIST_R,
        "left_upLeg_joint": HIP_L, "right_upLeg_joint": HIP_R,
        "left_leg_joint": KNEE_L, "right_leg_joint": KNEE_R,
        "left_foot_joint": ANKLE_L, "right_foot_joint": ANKLE_R,
    }
    for jname, idx in direct.items():
        if vis(idx):
            positions_2d[jname] = norm_2d(idx)
            w3 = world_3d(idx)
            if w3: positions_3d[jname] = w3

    return positions_3d, positions_2d


# ── Frame track compression ──

def build_frame_track(all_frames, all_pos_3d, all_pos_2d, fps, total_frames, duration_ms):
    """Build MovementFrameTrack with keyframe+delta compression."""
    keyframes, deltas = [], []
    last_kf_angles = None

    for i, frame in enumerate(all_frames):
        is_kf = (i % KEYFRAME_INTERVAL == 0) or i == 0 or i == len(all_frames) - 1
        if is_kf:
            keyframes.append({
                "frameIndex": frame["frameIndex"],
                "timestampMs": frame["timestampMs"],
                "angles": {k: round(v, 2) for k, v in frame["angles"].items()},
            })
            last_kf_angles = frame["angles"].copy()
        elif last_kf_angles:
            changes = {}
            for key, val in frame["angles"].items():
                if abs(val - last_kf_angles.get(key, 0)) > ANGLE_DELTA_THRESHOLD:
                    changes[key] = round(val, 2)
            if changes:
                deltas.append({"frameIndex": frame["frameIndex"], "timestampMs": frame["timestampMs"], "changes": changes})

    # Position keyframes/deltas (3D)
    pos_kf, pos_d = _compress_positions(all_pos_3d, POS_DELTA_THRESHOLD_3D)
    pos_kf_2d, pos_d_2d = _compress_positions(all_pos_2d, POS_DELTA_THRESHOLD_2D)

    return {
        "fps": fps,
        "totalFrames": total_frames,
        "durationMs": duration_ms,
        "keyframes": keyframes,
        "deltas": deltas or None,
        "positionKeyframes": pos_kf or None,
        "positionDeltas": pos_d or None,
        "positionKeyframes2D": pos_kf_2d or None,
        "positionDeltas2D": pos_d_2d or None,
    }


def _compress_positions(all_pos, threshold):
    kf_list, delta_list = [], []
    last_kf = None
    for i, pos in enumerate(all_pos):
        is_kf = (i % KEYFRAME_INTERVAL == 0) or i == 0 or i == len(all_pos) - 1
        if is_kf:
            kf_list.append({"f": pos["frameIndex"], "t": pos["timestampMs"], "p": {"p": pos["positions"]}})
            last_kf = pos["positions"].copy()
        elif last_kf:
            changed = {}
            for joint, coords in pos["positions"].items():
                prev = last_kf.get(joint)
                if prev and len(prev) == len(coords):
                    dist = np.sqrt(sum((a-b)**2 for a, b in zip(coords, prev)))
                    if dist > threshold:
                        changed[joint] = coords
                else:
                    changed[joint] = coords
            if changed:
                delta_list.append({"f": pos["frameIndex"], "t": pos["timestampMs"], "c": changed})
    return kf_list, delta_list


def build_body_channels(region_histories, fps, total_frames):
    """Build per-frame body channel waveform data."""
    channels = []
    num = len(region_histories["torso"])
    for i in range(num):
        channels.append({
            "torso": {"intensity": round(region_histories["torso"][i] / 50.0, 4), "onBeat": 0.0},
            "leftArm": {"intensity": round(region_histories["left_arm"][i] / 50.0, 4), "onBeat": 0.0},
            "rightArm": {"intensity": round(region_histories["right_arm"][i] / 50.0, 4), "onBeat": 0.0},
            "leftLeg": {"intensity": round(region_histories["left_leg"][i] / 50.0, 4), "onBeat": 0.0},
            "rightLeg": {"intensity": round(region_histories["right_leg"][i] / 50.0, 4), "onBeat": 0.0},
        })
    return channels


# ── Analysis summary builder ──

CLINICAL_NORMS = {
    'l_knee':     {'name': 'L Knee Flex',    'norm': (0, 135),  'unit': 'deg', 'convert': lambda v: 180-v},
    'r_knee':     {'name': 'R Knee Flex',    'norm': (0, 135),  'unit': 'deg', 'convert': lambda v: 180-v},
    'l_hip':      {'name': 'L Hip Flex',     'norm': (0, 120),  'unit': 'deg', 'convert': lambda v: 180-v},
    'r_hip':      {'name': 'R Hip Flex',     'norm': (0, 120),  'unit': 'deg', 'convert': lambda v: 180-v},
    'l_elbow':    {'name': 'L Elbow Flex',   'norm': (0, 145),  'unit': 'deg', 'convert': lambda v: 180-v},
    'r_elbow':    {'name': 'R Elbow Flex',   'norm': (0, 145),  'unit': 'deg', 'convert': lambda v: 180-v},
    'l_shoulder': {'name': 'L Shoulder Flex', 'norm': (0, 180), 'unit': 'deg', 'convert': lambda v: v},
    'r_shoulder': {'name': 'R Shoulder Flex', 'norm': (0, 180), 'unit': 'deg', 'convert': lambda v: v},
}

GAIT_NORMS = {
    'armSwingRatio': {'name': 'Arm Swing Ratio', 'norm': (0.6, 1.4)},
    'stepWidthCm':   {'name': 'Step Width',      'norm': (8, 15)},
}


def build_analysis_summary(detected_view, duration_sec, rom_tracker, gait_tracker, view_votes):
    """Build analysis summary JSON block."""
    # ROM
    rom = {}
    for key in ['l_knee', 'r_knee', 'l_hip', 'r_hip', 'l_elbow', 'r_elbow',
                 'l_shoulder', 'r_shoulder', 'l_ankle', 'r_ankle']:
        r = rom_tracker.get_rom(key)
        if r:
            mn, mx, cur = r
            rom[key] = {"min": round(mn, 1), "max": round(mx, 1), "range": round(mx - mn, 1)}

    # Gait metrics
    gait = {}
    arm_ratio = gait_tracker.arm_swing_ratio()
    gait["armSwingRatio"] = round(arm_ratio, 2)
    sw = gait_tracker.avg_step_width()
    if sw > 0:
        gait["stepWidthCm"] = round(sw * 100, 1)
    l_circ = gait_tracker.circumduction_index('l')
    r_circ = gait_tracker.circumduction_index('r')
    gait["circumductionL"] = round(l_circ, 1)
    gait["circumductionR"] = round(r_circ, 1)
    l_hike, r_hike, obliq = gait_tracker.hip_hiking()
    gait["hipHikeL"] = round(l_hike, 1)
    gait["hipHikeR"] = round(r_hike, 1)

    # Flags
    flags = []

    # ARM swing
    if abs(arm_ratio - 1.0) > 0.4:
        severity = "significant" if abs(arm_ratio - 1.0) > 0.8 else "mild"
        flags.append({"metric": "armSwingRatio", "value": round(arm_ratio, 2),
                       "severity": severity, "norm": "0.6-1.4"})

    # Step width
    if sw > 0 and sw * 100 > 15:
        flags.append({"metric": "stepWidthCm", "value": round(sw*100, 1),
                       "severity": "significant" if sw*100 > 20 else "mild", "norm": "8-15cm"})

    # ROM vs clinical norms
    for key, spec in CLINICAL_NORMS.items():
        r = rom_tracker.get_rom(key)
        if not r:
            continue
        mn, mx, _ = r
        measured_range = spec['convert'](mn)  # max flexion
        lo, hi = spec['norm']
        if measured_range > hi + 10:
            flags.append({"metric": key + "_rom", "value": round(measured_range, 1),
                           "severity": "significant", "norm": f"{lo}-{hi}"})

    # View confidence
    total_votes = view_votes.get('front', 0) + view_votes.get('side', 0)
    view_conf = view_votes.get(detected_view, 0) / max(total_votes, 1)

    return {
        "detectedView": detected_view,
        "viewConfidence": round(view_conf, 2),
        "durationSec": round(duration_sec, 1),
        "rom": rom,
        "gait": gait,
        "flags": flags,
    }


# ── Main processing loop ──

def process(input_path, output_video_path=None, include_channels=False):
    """Process a video file. Returns JSON output dict."""

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {input_path}", file=sys.stderr)
        sys.exit(1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_int = int(round(fps))

    print(f"Processing: {input_path} — {w}x{h} @ {fps:.1f}fps, {total} frames", file=sys.stderr)

    # Video writer (optional)
    out = None
    if output_video_path:
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (w, h))

    # Pose landmarker
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.6,
    )
    landmarker = PoseLandmarker.create_from_options(options)

    # Smoothers + trackers
    smoother = Smoother(alpha=SMOOTH_ALPHA)
    world_smoother = WorldSmoother(alpha=SMOOTH_ALPHA)
    rom_tracker = ROMTracker(window=int(fps * 3))
    gait_tracker = GaitTracker(window=int(fps * 3))

    prev_pts = {}
    histories = {r: deque(maxlen=int(fps * 8)) for r in REGION_LABELS}
    view_votes = {'front': 0, 'side': 0}
    stable_view = 'front'

    # JSON accumulators
    all_frames = []
    all_pos_3d = []
    all_pos_2d = []
    all_region_movements = {r: [] for r in REGION_LABELS}

    fidx = 0
    ts = 0.0
    dt = 1000.0 / fps

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_img, int(ts))
        ts += dt

        # 2D landmarks for drawing
        lm = result.pose_landmarks[0] if result.pose_landmarks and len(result.pose_landmarks) > 0 else None
        pts = smoother.update(lm, w, h)

        # 3D world landmarks for angle calculations
        world_lm = result.pose_world_landmarks[0] if result.pose_world_landmarks and len(result.pose_world_landmarks) > 0 else None
        world_pts = world_smoother.update(world_lm)

        # Movement waveforms
        mv = compute_movement(prev_pts, pts)
        for r in REGION_LABELS:
            histories[r].append(mv[r])
            all_region_movements[r].append(mv[r])
        prev_pts = pts

        # View detection (first 30 frames)
        v = detect_view(pts, w)
        if fidx < 30:
            view_votes[v] += 1
        elif fidx == 30:
            stable_view = 'front' if view_votes['front'] >= view_votes['side'] else 'side'
            print(f"  Detected view: {stable_view}", file=sys.stderr)
        if fidx >= 30:
            v = stable_view

        # Compute 3D angles
        angles = compute_joint_angles(world_pts, pts)
        rom_tracker.update(angles)
        gait_tracker.update(pts, world_pts)

        # Map to iOS angle keys
        ios_angles = map_to_ios_angles(angles)

        # ARKit positions
        pos_3d, pos_2d = compute_arkit_positions(smoother.state, world_lm, w, h)

        ts_int = int(round(ts - dt))

        # Accumulate for JSON
        all_frames.append({"frameIndex": fidx, "timestampMs": ts_int, "angles": ios_angles})
        if pos_3d:
            all_pos_3d.append({"frameIndex": fidx, "timestampMs": ts_int, "positions": pos_3d})
        if pos_2d:
            all_pos_2d.append({"frameIndex": fidx, "timestampMs": ts_int, "positions": pos_2d})

        # Draw rendered video
        if out:
            draw_skeleton(frame, pts)
            draw_joint_angles(frame, pts, angles, v)
            if v == 'front':
                draw_front_guides(frame, pts, angles)
                draw_frontal_panel(frame, angles, rom_tracker, gait_tracker, w)
            else:
                draw_sagittal_panel(frame, angles, rom_tracker, gait_tracker, w)
            draw_waveform(frame, histories, w, h)
            out.write(frame)

        fidx += 1
        if fidx % 100 == 0:
            pct = 100 * fidx // max(total, 1)
            print(f"  {fidx}/{total} ({pct}%)", file=sys.stderr)

    cap.release()
    landmarker.close()
    if out:
        out.release()
        print(f"Rendered video: {output_video_path} ({fidx} frames)", file=sys.stderr)

    duration_ms = int(round(ts - dt)) if fidx > 0 else 0
    duration_sec = duration_ms / 1000.0

    # Build compressed frame track
    frame_track = build_frame_track(all_frames, all_pos_3d, all_pos_2d, fps_int, fidx, duration_ms)

    # Build body channels
    channels = build_body_channels(all_region_movements, fps_int, fidx) if include_channels else None

    # Build analysis summary
    analysis = build_analysis_summary(stable_view, duration_sec, rom_tracker, gait_tracker, view_votes)

    output = {"frameTrack": frame_track, "analysis": analysis}
    if channels:
        output["bodyChannels"] = channels

    print(f"Done. {fidx} frames processed.", file=sys.stderr)
    return output


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 backend_processor.py <input.mp4> [--output-video <path>] [--channels]", file=sys.stderr)
        sys.exit(1)

    input_path = sys.argv[1]
    output_video = None
    include_channels = "--channels" in sys.argv

    # Parse --output-video <path>
    for i, arg in enumerate(sys.argv):
        if arg == "--output-video" and i + 1 < len(sys.argv):
            output_video = sys.argv[i + 1]

    if not os.path.exists(input_path):
        print(f"ERROR: File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    result = process(input_path, output_video, include_channels)

    json.dump(result, sys.stdout, separators=(",", ":"))
    print(file=sys.stderr)
    size = len(json.dumps(result, separators=(",", ":")))
    print(f"JSON output: {size} bytes", file=sys.stderr)


if __name__ == "__main__":
    main()
