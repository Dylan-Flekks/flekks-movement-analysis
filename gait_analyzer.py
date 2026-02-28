#!/usr/bin/env python3
"""
Flekks Gait Analyzer v2 — 3D Angles + Plane-Specific Metrics

Key improvements over v1:
- Uses pose_world_landmarks (3D coords in meters) for angle calculations
- angle_at_3d() for true 3D joint angles instead of 2D projection
- Ankle DF/PF uses FOOT_INDEX (31/32) not HEEL (29/30) for correct geometry
- Plane-specific panels: frontal metrics shown for frontal view, sagittal for side view
- 2D pts used only for drawing overlays; 3D world pts used for all angle math
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import sys
import math

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# ── COLORS (BGR) ──
ACCENT     = (166, 180, 64)
ACCENT_DIM = (115, 125, 45)
WHITE      = (240, 245, 240)
RED        = (70, 70, 230)
GREEN      = (100, 200, 100)
YELLOW     = (60, 200, 230)
GRAY       = (100, 100, 100)
PANEL_BG   = (20, 20, 20)
CYAN       = (200, 180, 50)
DARK_BG    = (15, 15, 15)
ORANGE     = (50, 140, 230)

REGION_COLORS = {
    'torso':     (120, 130, 140),
    'left_arm':  (166, 180, 64),
    'right_arm': (219, 135, 71),
    'left_leg':  (204, 104, 132),
    'right_leg': (58, 112, 196),
}
REGION_LABELS = ['torso', 'left_arm', 'right_arm', 'left_leg', 'right_leg']
REGION_DISPLAY = {'torso': 'TORSO', 'left_arm': 'L ARM', 'right_arm': 'R ARM',
                  'left_leg': 'L LEG', 'right_leg': 'R LEG'}
REGION_LANDMARKS = {
    'torso':     [11, 12, 23, 24],
    'left_arm':  [11, 13, 15],
    'right_arm': [12, 14, 16],
    'left_leg':  [23, 25, 27],
    'right_leg': [24, 26, 28],
}

# Landmarks
EAR_L, EAR_R = 7, 8
SHOULDER_L, SHOULDER_R = 11, 12
ELBOW_L, ELBOW_R = 13, 14
WRIST_L, WRIST_R = 15, 16
HIP_L, HIP_R = 23, 24
KNEE_L, KNEE_R = 25, 26
ANKLE_L, ANKLE_R = 27, 28
HEEL_L, HEEL_R = 29, 30
FOOT_L, FOOT_R = 31, 32

SKELETON = [
    (11,12),(11,23),(12,24),(23,24),
    (11,13),(13,15),(12,14),(14,16),
    (23,25),(25,27),(24,26),(26,28),
    (27,29),(28,30),(27,31),(28,32),
]
NOSE = 0
JOINTS = {0,7,8,11,12,13,14,15,16,23,24,25,26,27,28,29,30,31,32}
SMOOTH_ALPHA = 0.45

JOINT_TYPES = {
    'l_elbow': ('Elbow', 'FLEX', 'EXT'),
    'r_elbow': ('Elbow', 'FLEX', 'EXT'),
    'l_shoulder': ('Shoulder', 'FLEX', 'EXT'),
    'r_shoulder': ('Shoulder', 'FLEX', 'EXT'),
    'l_hip': ('Hip', 'FLEX', 'EXT'),
    'r_hip': ('Hip', 'FLEX', 'EXT'),
    'l_knee': ('Knee', 'FLEX', 'EXT'),
    'r_knee': ('Knee', 'FLEX', 'EXT'),
    'l_ankle': ('Ankle', 'DF', 'PF'),
    'r_ankle': ('Ankle', 'DF', 'PF'),
}

# Arc drawing uses 2D pts — ankle arcs now use FOOT_INDEX for visual
JOINT_ARCS = {
    'l_elbow':    (SHOULDER_L, ELBOW_L, WRIST_L),
    'r_elbow':    (SHOULDER_R, ELBOW_R, WRIST_R),
    'l_shoulder': (HIP_L, SHOULDER_L, ELBOW_L),
    'r_shoulder': (HIP_R, SHOULDER_R, ELBOW_R),
    'l_hip':      (SHOULDER_L, HIP_L, KNEE_L),
    'r_hip':      (SHOULDER_R, HIP_R, KNEE_R),
    'l_knee':     (HIP_L, KNEE_L, ANKLE_L),
    'r_knee':     (HIP_R, KNEE_R, ANKLE_R),
    'l_ankle':    (KNEE_L, ANKLE_L, FOOT_L),
    'r_ankle':    (KNEE_R, ANKLE_R, FOOT_R),
}

# ROM window (frames) for min/max tracking
ROM_WINDOW = 90  # ~3 seconds at 30fps


# ── SMOOTHERS ──

class Smoother:
    """Smooth 2D pose_landmarks for drawing overlays."""
    def __init__(self, alpha=SMOOTH_ALPHA, n=33):
        self.alpha, self.n, self.state = alpha, n, None
    def update(self, landmarks, w, h):
        if landmarks is None or len(landmarks) == 0:
            if self.state is not None: self.state[:, 3] *= 0.85
            return self._pts(w, h)
        raw = np.zeros((self.n, 4), dtype=np.float64)
        for i, lm in enumerate(landmarks):
            if i >= self.n: break
            raw[i] = [lm.x, lm.y, lm.z, lm.visibility]
        if self.state is None: self.state = raw.copy()
        else:
            self.state[:,:3] = self.alpha*self.state[:,:3]+(1-self.alpha)*raw[:,:3]
            self.state[:,3] = raw[:,3]
        return self._pts(w, h)
    def _pts(self, w, h):
        if self.state is None: return {}
        return {i:(int(self.state[i,0]*w),int(self.state[i,1]*h))
                for i in range(self.n) if self.state[i,3]>0.3}


class WorldSmoother:
    """Smooth 3D pose_world_landmarks (meters, pelvis-centered) for angle calculations."""
    def __init__(self, alpha=SMOOTH_ALPHA, n=33):
        self.alpha, self.n, self.state = alpha, n, None
    def update(self, landmarks):
        if landmarks is None or len(landmarks) == 0:
            if self.state is not None: self.state[:, 3] *= 0.85
            return self._pts()
        raw = np.zeros((self.n, 4), dtype=np.float64)
        for i, lm in enumerate(landmarks):
            if i >= self.n: break
            raw[i] = [lm.x, lm.y, lm.z, lm.visibility]
        if self.state is None: self.state = raw.copy()
        else:
            self.state[:,:3] = self.alpha*self.state[:,:3]+(1-self.alpha)*raw[:,:3]
            self.state[:,3] = raw[:,3]
        return self._pts()
    def _pts(self):
        if self.state is None: return {}
        return {i:(self.state[i,0], self.state[i,1], self.state[i,2])
                for i in range(self.n) if self.state[i,3]>0.3}


# ── TRACKERS ──

class ROMTracker:
    """Track min/max angles over a rolling window for ROM display."""
    def __init__(self, window=ROM_WINDOW):
        self.window = window
        self.history = {}

    def update(self, angles):
        for key in ['l_knee', 'r_knee', 'l_hip', 'r_hip', 'l_ankle', 'r_ankle',
                     'l_elbow', 'r_elbow', 'l_shoulder', 'r_shoulder']:
            if key in angles:
                if key not in self.history:
                    self.history[key] = deque(maxlen=self.window)
                self.history[key].append(angles[key])

    def get_rom(self, key):
        if key not in self.history or len(self.history[key]) < 5:
            return None
        vals = list(self.history[key])
        return (min(vals), max(vals), vals[-1])


class GaitTracker:
    """Track hemiparesis-specific gait indices from 2D pixel positions."""
    def __init__(self, window=ROM_WINDOW):
        self.window = window
        self.l_ankle_lat = deque(maxlen=window)
        self.r_ankle_lat = deque(maxlen=window)
        self.l_ankle_y = deque(maxlen=window)
        self.r_ankle_y = deque(maxlen=window)
        self.l_hip_y = deque(maxlen=window)
        self.r_hip_y = deque(maxlen=window)
        self.l_arm_swing = deque(maxlen=window)
        self.r_arm_swing = deque(maxlen=window)
        self.prev_l_wrist = None
        self.prev_r_wrist = None
        # Step width from world coords
        self.step_widths = deque(maxlen=window)

    def update(self, pts, world_pts=None):
        if HIP_L in pts and ANKLE_L in pts:
            lat = pts[ANKLE_L][0] - pts[HIP_L][0]
            self.l_ankle_lat.append(lat)
            self.l_ankle_y.append(pts[ANKLE_L][1])
        if HIP_R in pts and ANKLE_R in pts:
            lat = pts[ANKLE_R][0] - pts[HIP_R][0]
            self.r_ankle_lat.append(lat)
            self.r_ankle_y.append(pts[ANKLE_R][1])

        if HIP_L in pts:
            self.l_hip_y.append(pts[HIP_L][1])
        if HIP_R in pts:
            self.r_hip_y.append(pts[HIP_R][1])

        if WRIST_L in pts:
            if self.prev_l_wrist is not None:
                dy = abs(pts[WRIST_L][1] - self.prev_l_wrist[1])
                self.l_arm_swing.append(dy)
            self.prev_l_wrist = pts[WRIST_L]
        if WRIST_R in pts:
            if self.prev_r_wrist is not None:
                dy = abs(pts[WRIST_R][1] - self.prev_r_wrist[1])
                self.r_arm_swing.append(dy)
            self.prev_r_wrist = pts[WRIST_R]

        # Step width from 3D world coords (lateral distance between ankles)
        if world_pts and ANKLE_L in world_pts and ANKLE_R in world_pts:
            lx = world_pts[ANKLE_L][0]
            rx = world_pts[ANKLE_R][0]
            self.step_widths.append(abs(lx - rx))

    def circumduction_index(self, side):
        lat = self.l_ankle_lat if side == 'l' else self.r_ankle_lat
        if len(lat) < 10:
            return 0
        vals = list(lat)
        return max(vals) - min(vals)

    def hip_hiking(self):
        if len(self.l_hip_y) < 10 or len(self.r_hip_y) < 10:
            return 0, 0, 0
        l_range = max(self.l_hip_y) - min(self.l_hip_y)
        r_range = max(self.r_hip_y) - min(self.r_hip_y)
        l_avg = np.mean(list(self.l_hip_y))
        r_avg = np.mean(list(self.r_hip_y))
        return l_range, r_range, abs(l_avg - r_avg)

    def arm_swing_ratio(self):
        if len(self.l_arm_swing) < 10 or len(self.r_arm_swing) < 10:
            return 1.0
        l_avg = np.mean(list(self.l_arm_swing))
        r_avg = np.mean(list(self.r_arm_swing))
        if max(l_avg, r_avg) < 1:
            return 1.0
        return l_avg / max(r_avg, 0.1)

    def avg_step_width(self):
        if len(self.step_widths) < 10:
            return 0.0
        return np.mean(list(self.step_widths))


# ── GEOMETRY ──

def mid(a, b): return ((a[0]+b[0])//2,(a[1]+b[1])//2)


def angle_at(p1, v, p2):
    """2D angle at vertex v between rays v->p1 and v->p2."""
    v1=np.array([p1[0]-v[0],p1[1]-v[1]],dtype=np.float64)
    v2=np.array([p2[0]-v[0],p2[1]-v[1]],dtype=np.float64)
    n1,n2=np.linalg.norm(v1),np.linalg.norm(v2)
    if n1<1 or n2<1: return 180.0
    c=np.dot(v1,v2)/(n1*n2)
    return np.degrees(np.arccos(np.clip(c,-1,1)))


def angle_at_3d(p1, v, p2):
    """3D angle at vertex v between rays v->p1 and v->p2. Points are (x,y,z) tuples."""
    v1 = np.array([p1[0]-v[0], p1[1]-v[1], p1[2]-v[2]], dtype=np.float64)
    v2 = np.array([p2[0]-v[0], p2[1]-v[1], p2[2]-v[2]], dtype=np.float64)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0
    c = np.dot(v1, v2) / (n1 * n2)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def compute_ankle_df_pf_3d(knee, ankle, foot_index):
    """
    Compute ankle DF/PF from 3D world landmarks using FOOT_INDEX.
    angle_at_3d(KNEE, ANKLE, FOOT_INDEX) ~ 90 deg at neutral.
    DF = 90 - angle (positive, foot toward shin)
    PF = angle - 90 (positive, foot pointing away)
    Returns: positive = DF, negative = PF, 0 = neutral
    """
    angle = angle_at_3d(knee, ankle, foot_index)
    # When angle < 90, foot is dorsiflexed (toward shin)
    # When angle > 90, foot is plantarflexed (away from shin)
    return 90.0 - angle


def angle_from_vertical_3d(top, bottom):
    """Angle of segment from vertical (y-axis) in 3D. Returns degrees."""
    dx = top[0] - bottom[0]
    dy = top[1] - bottom[1]
    dz = top[2] - bottom[2]
    seg_len = math.sqrt(dx*dx + dy*dy + dz*dz)
    if seg_len < 1e-6:
        return 0.0
    # vertical is (0, -1, 0) in world coords (y points down in mediapipe world)
    # cos(angle) = dot(seg, vertical) / |seg|
    # seg normalized dot (0,-1,0) = -dy/seg_len
    cos_a = -dy / seg_len
    return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))



def normalize_vec(v):
    """Normalize a 3D vector. Returns zero vector if magnitude is too small."""
    n = np.linalg.norm(v)
    if n < 1e-6:
        return np.zeros(3)
    return v / n


def trunk_axis(world_pts):
    """Trunk vertical axis: mid-hip -> mid-shoulder, normalized. Returns 3D unit vector pointing up along trunk."""
    wp = world_pts
    if SHOULDER_L not in wp or SHOULDER_R not in wp or HIP_L not in wp or HIP_R not in wp:
        return None
    mid_s = np.array([(wp[SHOULDER_L][0]+wp[SHOULDER_R][0])/2,
                      (wp[SHOULDER_L][1]+wp[SHOULDER_R][1])/2,
                      (wp[SHOULDER_L][2]+wp[SHOULDER_R][2])/2])
    mid_h = np.array([(wp[HIP_L][0]+wp[HIP_R][0])/2,
                      (wp[HIP_L][1]+wp[HIP_R][1])/2,
                      (wp[HIP_L][2]+wp[HIP_R][2])/2])
    axis = mid_s - mid_h  # points from hips up to shoulders
    return normalize_vec(axis)


def angle_between_vectors(v1, v2):
    """Angle in degrees between two 3D vectors."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    c = np.dot(v1, v2) / (n1 * n2)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def gonio_shoulder_flex(world_pts, side):
    """Shoulder flexion: angle between trunk axis (pointing down = arm-at-side) and upper arm.
    0° = arm at side, 90° = arm horizontal, 180° = arm overhead."""
    wp = world_pts
    trunk_up = trunk_axis(wp)
    if trunk_up is None:
        return None
    sh = SHOULDER_L if side == 'l' else SHOULDER_R
    el = ELBOW_L if side == 'l' else ELBOW_R
    if sh not in wp or el not in wp:
        return None
    arm = np.array(wp[el]) - np.array(wp[sh])  # shoulder -> elbow
    # Arm at side = parallel to trunk pointing down = -trunk_up
    return angle_between_vectors(-trunk_up, arm)


def gonio_hip_abd(world_pts, side):
    """Hip ABD: angle between pelvis midline (downward along trunk) and femur (hip->knee).
    Positive = abducted (knee lateral to midline), negative = adducted (knee medial).
    Normal ABD ROM: 0-45°."""
    wp = world_pts
    trunk_up = trunk_axis(wp)
    if trunk_up is None:
        return None
    hip_i = HIP_L if side == 'l' else HIP_R
    knee_i = KNEE_L if side == 'l' else KNEE_R
    hip_other = HIP_R if side == 'l' else HIP_L
    if hip_i not in wp or knee_i not in wp or hip_other not in wp:
        return None
    femur = np.array(wp[knee_i]) - np.array(wp[hip_i])  # hip -> knee
    pelvis_down = -trunk_up  # trunk axis pointing downward
    angle = angle_between_vectors(pelvis_down, femur)
    # Determine ABD vs ADD: is knee lateral (away from midline) or medial?
    # Hip-to-hip vector points from this hip toward the other hip (toward midline)
    toward_midline = np.array(wp[hip_other]) - np.array(wp[hip_i])
    # Femur component along midline direction: positive = toward midline = ADDucting
    femur_toward_mid = np.dot(femur, toward_midline)
    if femur_toward_mid > 0:
        return angle   # ABD (positive) — knee lateral, away from midline
    else:
        return -angle  # ADD (negative) — knee medial, toward midline


def gonio_shoulder_abd(world_pts, side):
    """Shoulder ABD: angle between trunk axis (pointing down) and upper arm (shoulder->elbow).
    0° = arm at side, 90° = arm horizontal."""
    wp = world_pts
    trunk_up = trunk_axis(wp)
    if trunk_up is None:
        return None
    sh = SHOULDER_L if side == 'l' else SHOULDER_R
    el = ELBOW_L if side == 'l' else ELBOW_R
    if sh not in wp or el not in wp:
        return None
    arm = np.array(wp[el]) - np.array(wp[sh])
    return angle_between_vectors(-trunk_up, arm)


def detect_view(pts, frame_w=1080):
    """Detect frontal vs sagittal view using relative shoulder/hip separation."""
    if SHOULDER_L not in pts or SHOULDER_R not in pts: return 'front'
    dx=abs(pts[SHOULDER_L][0]-pts[SHOULDER_R][0])
    hw=abs(pts[HIP_L][0]-pts[HIP_R][0]) if HIP_L in pts and HIP_R in pts else 0
    avg_sep = (dx+hw)/2
    # Use 15% of frame width as threshold — accounts for different resolutions
    threshold = frame_w * 0.15
    return 'side' if avg_sep < threshold else 'front'


def compute_movement(prev, curr):
    result = {}
    for region, indices in REGION_LANDMARKS.items():
        total, count = 0.0, 0
        for i in indices:
            if i in curr and i in prev:
                dx=curr[i][0]-prev[i][0]; dy=curr[i][1]-prev[i][1]
                total+=np.sqrt(dx*dx+dy*dy); count+=1
        result[region]=total/max(count,1)
    return result


def smooth_wave(vals, k=11):
    if len(vals)<k: return vals
    ker=cv2.getGaussianKernel(k,0).flatten(); ker/=ker.sum()
    return np.convolve(np.pad(np.array(vals,dtype=np.float64),k//2,mode='edge'),ker,mode='valid').tolist()


def norm_color(val, lo, hi):
    if lo<=val<=hi: return GREEN
    if (lo-5)<=val<=(hi+5): return YELLOW
    return RED


# ── ANGLE COMPUTATION (3D world landmarks) ──

def compute_joint_angles(world_pts, pts_2d):
    """
    Compute all joint angles from 3D world landmarks.
    pts_2d is used only for 2D-specific metrics (hip shift, FPPA visual guides).
    """
    a = {}
    wp = world_pts

    # ── Sagittal angles (meaningful in side view, computed from 3D) ──

    # Elbow FLEX/EXT
    if SHOULDER_L in wp and ELBOW_L in wp and WRIST_L in wp:
        a['l_elbow'] = angle_at_3d(wp[SHOULDER_L], wp[ELBOW_L], wp[WRIST_L])
    if SHOULDER_R in wp and ELBOW_R in wp and WRIST_R in wp:
        a['r_elbow'] = angle_at_3d(wp[SHOULDER_R], wp[ELBOW_R], wp[WRIST_R])

    # Shoulder FLEX/EXT — goniometric: angle between trunk axis and upper arm
    # 0° = arm at side, 90° = arm horizontal, 180° = overhead
    l_sh_flex = gonio_shoulder_flex(wp, 'l')
    if l_sh_flex is not None:
        a['l_shoulder'] = l_sh_flex
    r_sh_flex = gonio_shoulder_flex(wp, 'r')
    if r_sh_flex is not None:
        a['r_shoulder'] = r_sh_flex

    # Hip FLEX/EXT
    if SHOULDER_L in wp and HIP_L in wp and KNEE_L in wp:
        a['l_hip'] = angle_at_3d(wp[SHOULDER_L], wp[HIP_L], wp[KNEE_L])
    if SHOULDER_R in wp and HIP_R in wp and KNEE_R in wp:
        a['r_hip'] = angle_at_3d(wp[SHOULDER_R], wp[HIP_R], wp[KNEE_R])

    # Knee FLEX/EXT
    if HIP_L in wp and KNEE_L in wp and ANKLE_L in wp:
        a['l_knee'] = angle_at_3d(wp[HIP_L], wp[KNEE_L], wp[ANKLE_L])
    if HIP_R in wp and KNEE_R in wp and ANKLE_R in wp:
        a['r_knee'] = angle_at_3d(wp[HIP_R], wp[KNEE_R], wp[ANKLE_R])

    # Ankle DF/PF — using FOOT_INDEX (31/32) for correct geometry
    if KNEE_L in wp and ANKLE_L in wp and FOOT_L in wp:
        a['l_ankle_dfpf'] = compute_ankle_df_pf_3d(wp[KNEE_L], wp[ANKLE_L], wp[FOOT_L])
        a['l_ankle'] = angle_at_3d(wp[KNEE_L], wp[ANKLE_L], wp[FOOT_L])
    if KNEE_R in wp and ANKLE_R in wp and FOOT_R in wp:
        a['r_ankle_dfpf'] = compute_ankle_df_pf_3d(wp[KNEE_R], wp[ANKLE_R], wp[FOOT_R])
        a['r_ankle'] = angle_at_3d(wp[KNEE_R], wp[ANKLE_R], wp[FOOT_R])

    # Trunk lean (3D: angle of trunk from vertical)
    if SHOULDER_L in wp and SHOULDER_R in wp and HIP_L in wp and HIP_R in wp:
        mid_s = ((wp[SHOULDER_L][0]+wp[SHOULDER_R][0])/2,
                 (wp[SHOULDER_L][1]+wp[SHOULDER_R][1])/2,
                 (wp[SHOULDER_L][2]+wp[SHOULDER_R][2])/2)
        mid_h = ((wp[HIP_L][0]+wp[HIP_R][0])/2,
                 (wp[HIP_L][1]+wp[HIP_R][1])/2,
                 (wp[HIP_L][2]+wp[HIP_R][2])/2)
        a['trunk'] = angle_from_vertical_3d(mid_s, mid_h)

    # CVA (craniovertebral angle) — 3D
    # Angle at C7 (approx mid-shoulder) between vertical and line to ear
    if EAR_L in wp and SHOULDER_L in wp and SHOULDER_R in wp:
        c7 = ((wp[SHOULDER_L][0]+wp[SHOULDER_R][0])/2,
              (wp[SHOULDER_L][1]+wp[SHOULDER_R][1])/2,
              (wp[SHOULDER_L][2]+wp[SHOULDER_R][2])/2)
        ear = wp[EAR_L]
        # Vector from C7 to ear
        dx = ear[0] - c7[0]
        dy = ear[1] - c7[1]
        dz = ear[2] - c7[2]
        seg_len = math.sqrt(dx*dx + dy*dy + dz*dz)
        if seg_len > 1e-6:
            # CVA = angle between horizontal and C7->ear line
            # In world coords, horizontal plane is x-z, vertical is -y
            horiz_dist = math.sqrt(dx*dx + dz*dz)
            a['cva'] = np.degrees(np.arctan2(-dy, horiz_dist))
            # Ideal CVA ~50-60 deg. Lower = forward head posture

    # ── Frontal plane angles (meaningful in front view) ──

    # Hip ABD/ADD — goniometric: angle between pelvis midline (trunk down) and femur
    # 0° = legs together along trunk axis, positive = abducted
    l_hip_abd = gonio_hip_abd(wp, 'l')
    if l_hip_abd is not None:
        a['l_hip_abd'] = l_hip_abd
    r_hip_abd = gonio_hip_abd(wp, 'r')
    if r_hip_abd is not None:
        a['r_hip_abd'] = r_hip_abd

    # Shoulder ABD — goniometric: angle between trunk axis and upper arm
    # 0° = arm at side, 90° = arm horizontal
    l_sh_abd = gonio_shoulder_abd(wp, 'l')
    if l_sh_abd is not None:
        a['l_sh_abd'] = l_sh_abd
    r_sh_abd = gonio_shoulder_abd(wp, 'r')
    if r_sh_abd is not None:
        a['r_sh_abd'] = r_sh_abd

    # Ankle eversion/inversion — frontal plane angle of foot relative to horizontal
    # Uses HEEL→FOOT_INDEX vector projected into frontal (x-y) plane
    # Eversion = foot sole faces outward (positive), Inversion = inward (negative)
    for heel_i, foot_i, prefix in [(HEEL_L, FOOT_L, 'l'), (HEEL_R, FOOT_R, 'r')]:
        if heel_i in wp and foot_i in wp:
            fx = wp[foot_i][0] - wp[heel_i][0]  # medial-lateral
            fy = wp[foot_i][1] - wp[heel_i][1]  # vertical
            foot_len = math.sqrt(fx*fx + fy*fy)
            if foot_len > 1e-6:
                # Angle from horizontal
                tilt = np.degrees(np.arctan2(-fy, abs(fx)))
                # Eversion = foot tilts outward: L foot tilts left (+x), R foot tilts right (-x)
                if prefix == 'l':
                    a[f'{prefix}_ev_inv'] = tilt if fx > 0 else -tilt
                else:
                    a[f'{prefix}_ev_inv'] = tilt if fx < 0 else -tilt

    # ── Asymmetries ──
    for joint in ['elbow','shoulder','hip','knee','ankle']:
        lk, rk = f'l_{joint}', f'r_{joint}'
        if lk in a and rk in a:
            a[f'{joint}_asym'] = abs(a[lk] - a[rk])

    # ── 2D-based metrics (for frontal visual overlays) ──
    p = pts_2d

    # FPPA (knee valgus/varus) — 2D frontal projection
    for hip_i,knee_i,ankle_i,prefix in [(HIP_L,KNEE_L,ANKLE_L,'l'),(HIP_R,KNEE_R,ANKLE_R,'r')]:
        if hip_i in p and knee_i in p and ankle_i in p:
            v1=np.array([p[hip_i][0]-p[knee_i][0],p[hip_i][1]-p[knee_i][1]],dtype=np.float64)
            v2=np.array([p[ankle_i][0]-p[knee_i][0],p[ankle_i][1]-p[knee_i][1]],dtype=np.float64)
            n1,n2=np.linalg.norm(v1),np.linalg.norm(v2)
            if n1>1 and n2>1:
                c=np.dot(v1,v2)/(n1*n2)
                dev=180.0-np.degrees(np.arccos(np.clip(c,-1,1)))
                cross=v1[0]*v2[1]-v1[1]*v2[0]
                if prefix=='l':
                    a[f'{prefix}_fppa']=dev if cross>0 else -dev
                else:
                    a[f'{prefix}_fppa']=-dev if cross<0 else dev

    # Hip shift (2D)
    ms=mid(p[SHOULDER_L],p[SHOULDER_R]) if SHOULDER_L in p and SHOULDER_R in p else None
    mh=mid(p[HIP_L],p[HIP_R]) if HIP_L in p and HIP_R in p else None
    if ms and mh:
        sw=max(abs(p[SHOULDER_R][0]-p[SHOULDER_L][0]),1)
        a['hip_shift']=(mh[0]-ms[0])/sw*100
    a['mid_s']=ms; a['mid_h']=mh

    # Shoulder tilt (2D)
    if SHOULDER_L in p and SHOULDER_R in p:
        dy=p[SHOULDER_R][1]-p[SHOULDER_L][1]
        dx=max(abs(p[SHOULDER_R][0]-p[SHOULDER_L][0]),1)
        a['sh_tilt']=np.degrees(np.arctan2(abs(dy),dx))

    # Pelvic obliquity (2D pixel difference)
    if HIP_L in p and HIP_R in p:
        a['pelv_obliq'] = p[HIP_L][1] - p[HIP_R][1]

    return a


# ── DRAWING ──

def draw_skeleton(f, pts):
    for a,b in SKELETON:
        if a in pts and b in pts:
            cv2.line(f,pts[a],pts[b],ACCENT_DIM,6,cv2.LINE_AA)
    for a,b in SKELETON:
        if a in pts and b in pts:
            cv2.line(f,pts[a],pts[b],ACCENT,2,cv2.LINE_AA)

    for i in JOINTS:
        if i in pts:
            cv2.circle(f,pts[i],6,ACCENT_DIM,-1,cv2.LINE_AA)
            cv2.circle(f,pts[i],4,ACCENT,-1,cv2.LINE_AA)
            cv2.circle(f,pts[i],2,WHITE,-1,cv2.LINE_AA)


def draw_arc(f, vertex, p1, p2, angle_deg, color, radius=28, thickness=2):
    vx, vy = vertex
    dx1, dy1 = float(p1[0]-vx), float(p1[1]-vy)
    dx2, dy2 = float(p2[0]-vx), float(p2[1]-vy)
    n1 = math.sqrt(dx1*dx1 + dy1*dy1)
    n2 = math.sqrt(dx2*dx2 + dy2*dy2)
    if n1 < 1 or n2 < 1:
        return
    ang1 = math.degrees(math.atan2(-dy1, dx1))
    ang2 = math.degrees(math.atan2(-dy2, dx2))
    diff = (ang2 - ang1) % 360
    if diff > 180:
        start_angle = ang2
        sweep = 360 - diff
    else:
        start_angle = ang1
        sweep = diff
    cv2.ellipse(f, (vx, vy), (radius, radius), 0,
                -start_angle - sweep, -start_angle,
                color, thickness, cv2.LINE_AA)
    for ang in [ang1, ang2]:
        rad = math.radians(ang)
        ex = int(vx + (radius - 4) * math.cos(rad))
        ey = int(vy - (radius - 4) * math.sin(rad))
        ex2 = int(vx + (radius + 4) * math.cos(rad))
        ey2 = int(vy - (radius + 4) * math.sin(rad))
        cv2.line(f, (ex, ey), (ex2, ey2), color, 1, cv2.LINE_AA)


def get_ankle_label(dfpf_val):
    if dfpf_val > 3:
        return f"DF {dfpf_val:.0f}\u00b0"
    elif dfpf_val < -3:
        return f"PF {abs(dfpf_val):.0f}\u00b0"
    else:
        return "NEUT"


def get_flexion_label(key, angle_deg, angles):
    if key.endswith('ankle'):
        dfpf_key = key + '_dfpf'
        if dfpf_key in angles:
            return get_ankle_label(angles[dfpf_key])
        return f"{angle_deg:.0f}\u00b0"

    if key not in JOINT_TYPES:
        return f"{angle_deg:.0f}\u00b0"

    joint_name, flex_label, ext_label = JOINT_TYPES[key]

    # Shoulder uses goniometric convention: stored value IS the goniometric angle
    # 0° = arm at side (neutral), positive = flexion
    if 'shoulder' in key:
        if angle_deg < 5:
            return "NEUT"
        return f"{flex_label} {angle_deg:.0f}\u00b0"

    # Knee, hip, elbow: stored as included angle (180° = full extension)
    # Convert: goniometric flexion = 180 - included angle
    if angle_deg >= 175:
        return f"{ext_label}"
    else:
        return f"{flex_label} {180-angle_deg:.0f}\u00b0"


def draw_joint_angles(f, pts, angles, view):
    """Draw arc indicators and labels on joints appropriate to the view plane."""
    # Sagittal plane: knee flex/ext, hip flex/ext, ankle DF/PF, shoulder flex/ext (no elbow)
    sagittal_arcs = {'l_knee', 'r_knee', 'l_hip', 'r_hip', 'l_ankle', 'r_ankle',
                     'l_shoulder', 'r_shoulder'}
    # Frontal plane: elbow flex/ext is visible but nothing else from JOINT_ARCS
    # (frontal metrics like ABD/ADD, valgus, eversion are shown in panel + FPPA labels)
    frontal_arcs = {'l_elbow', 'r_elbow'}

    show_keys = sagittal_arcs if view == 'side' else frontal_arcs

    for key, (p1_i, vertex_i, p2_i) in JOINT_ARCS.items():
        if key not in angles:
            continue
        if key not in show_keys:
            continue
        if p1_i not in pts or vertex_i not in pts or p2_i not in pts:
            continue

        val = angles[key]
        vertex = pts[vertex_i]
        p1 = pts[p1_i]
        p2 = pts[p2_i]

        joint_base = key[2:]
        asym = angles.get(f'{joint_base}_asym', 0)

        if asym > 12:
            color = RED; label_color = RED
        elif asym > 6:
            color = ORANGE; label_color = YELLOW
        else:
            color = CYAN; label_color = CYAN

        if val > 170:
            arc_color = tuple(max(c//2, 0) for c in color)
            radius = 20
        else:
            arc_color = color
            radius = min(30, max(18, int(40 * (180 - val) / 180)))

        draw_arc(f, vertex, p1, p2, val, arc_color, radius=radius, thickness=2)

        dx1 = float(p1[0]-vertex[0])
        dy1 = float(p1[1]-vertex[1])
        dx2 = float(p2[0]-vertex[0])
        dy2 = float(p2[1]-vertex[1])
        n1 = max(math.sqrt(dx1*dx1+dy1*dy1), 1)
        n2 = max(math.sqrt(dx2*dx2+dy2*dy2), 1)
        bx = dx1/n1 + dx2/n2
        by = dy1/n1 + dy2/n2
        bn = max(math.sqrt(bx*bx+by*by), 0.01)
        bx /= bn; by /= bn

        label = get_flexion_label(key, val, angles)
        label_offset = radius + 18

        tx = int(vertex[0] + bx * label_offset)
        ty = int(vertex[1] + by * label_offset)
        tx = max(5, min(tx, f.shape[1]-120))
        ty = max(16, min(ty, f.shape[0]-5))

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)

        pad_x, pad_y = 6, 4
        x1, y1 = tx - pad_x, ty - th - pad_y
        x2, y2 = tx + tw + pad_x, ty + pad_y + 1

        overlay = f.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), DARK_BG, -1)
        cv2.addWeighted(overlay, 0.8, f, 0.2, 0, f)

        border_color = tuple(max(c//2, 0) for c in label_color)
        cv2.rectangle(f, (x1, y1), (x2, y2), border_color, 1, cv2.LINE_AA)
        cv2.putText(f, label, (tx, ty), font, font_scale, label_color, 2, cv2.LINE_AA)

        arc_edge_x = int(vertex[0] + bx * radius)
        arc_edge_y = int(vertex[1] + by * radius)
        label_near_x = int(vertex[0] + bx * (label_offset - 6))
        label_near_y = int(vertex[1] + by * (label_offset - 6))
        cv2.line(f, (arc_edge_x, arc_edge_y), (label_near_x, label_near_y),
                 border_color, 1, cv2.LINE_AA)

    # FPPA labels at knees (frontal view only)
    if view == 'front':
        for prefix, knee_i, ox in [('l', KNEE_L, -70), ('r', KNEE_R, 20)]:
            fppa_key = f'{prefix}_fppa'
            if fppa_key in angles and knee_i in pts:
                val = angles[fppa_key]
                if abs(val) < 3:
                    ktype = 'NEUTRAL'
                elif val > 0:
                    ktype = f'VALGUS {abs(val):.0f}\u00b0'
                else:
                    ktype = f'VARUS {abs(val):.0f}\u00b0'
                color = GREEN if abs(val) < 8 else (YELLOW if abs(val) < 12 else RED)
                pt = pts[knee_i]
                tx, ty = pt[0]+ox, pt[1]+28
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.45
                (tw, th), _ = cv2.getTextSize(ktype, font, font_scale, 1)
                pad_x, pad_y = 5, 3
                overlay = f.copy()
                cv2.rectangle(overlay, (tx-pad_x, ty-th-pad_y), (tx+tw+pad_x, ty+pad_y+1), DARK_BG, -1)
                cv2.addWeighted(overlay, 0.8, f, 0.2, 0, f)
                border = tuple(max(c//2, 0) for c in color)
                cv2.rectangle(f, (tx-pad_x, ty-th-pad_y), (tx+tw+pad_x, ty+pad_y+1), border, 1, cv2.LINE_AA)
                cv2.putText(f, ktype, (tx, ty), font, font_scale, color, 2, cv2.LINE_AA)

    # Hip ABD/ADD arcs at hips (frontal view only)
    # Arc from vertical (trunk down) reference to femur direction
    if view == 'front':
        for prefix, hip_i, knee_i in [('l', HIP_L, KNEE_L), ('r', HIP_R, KNEE_R)]:
            abd_key = f'{prefix}_hip_abd'
            if abd_key not in angles or hip_i not in pts or knee_i not in pts:
                continue
            val = angles[abd_key]  # signed: positive=ABD, negative=ADD
            abs_val = abs(val)
            if abs_val < 2:
                continue  # skip tiny angles

            hip_pt = pts[hip_i]
            knee_pt = pts[knee_i]

            # Color based on clinical norms: ABD 0-45° normal, ADD typically < 30°
            # Flag asymmetry between sides
            other_key = 'r_hip_abd' if prefix == 'l' else 'l_hip_abd'
            other_val = abs(angles.get(other_key, 0))
            asym = abs(abs_val - other_val)
            if asym > 10:
                color = RED; label_color = RED
            elif asym > 5:
                color = ORANGE; label_color = YELLOW
            else:
                color = CYAN; label_color = CYAN

            # Create a virtual "vertical down" point below the hip for the arc reference
            femur_len = max(math.sqrt((knee_pt[0]-hip_pt[0])**2 + (knee_pt[1]-hip_pt[1])**2), 1)
            vert_pt = (hip_pt[0], int(hip_pt[1] + femur_len * 0.6))

            # Draw the arc between vertical reference and femur
            radius = min(32, max(20, int(femur_len * 0.2)))
            draw_arc(f, hip_pt, vert_pt, knee_pt, abs_val, color, radius=radius, thickness=2)

            # Draw thin vertical reference line
            cv2.line(f, hip_pt, vert_pt, (50, 50, 50), 1, cv2.LINE_AA)

            # Label: ABD/NEUT/ADD
            if val > 3:
                label = f"ABD {abs_val:.0f}\u00b0"
            elif val < -3:
                label = f"ADD {abs_val:.0f}\u00b0"
            else:
                label = "NEUT"

            # Position label well outside body (left hip -> far left, right hip -> far right)
            label_dir = -1 if prefix == 'l' else 1
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)

            outward_offset = int(femur_len * 0.55)  # scale with body size
            tx = hip_pt[0] + label_dir * outward_offset
            if prefix == 'l':
                tx = tx - tw  # right-align for left hip
            ty = hip_pt[1] + 4

            # Clamp to frame
            tx = max(5, min(tx, f.shape[1] - tw - 5))
            ty = max(16, min(ty, f.shape[0] - 5))

            # Background pill
            pad_x, pad_y = 5, 3
            x1, y1 = tx - pad_x, ty - th - pad_y
            x2, y2 = tx + tw + pad_x, ty + pad_y + 1
            overlay = f.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), DARK_BG, -1)
            cv2.addWeighted(overlay, 0.8, f, 0.2, 0, f)
            border_color = tuple(max(c//2, 0) for c in label_color)
            cv2.rectangle(f, (x1, y1), (x2, y2), border_color, 1, cv2.LINE_AA)
            cv2.putText(f, label, (tx, ty), font, font_scale, label_color, 2, cv2.LINE_AA)

            # Leader line from arc outward to label
            leader_start_x = hip_pt[0] + label_dir * (radius + 2)
            leader_start_y = hip_pt[1]
            leader_end_x = tx if prefix == 'r' else tx + tw
            leader_end_y = ty - th // 2
            cv2.line(f, (leader_start_x, leader_start_y), (leader_end_x, leader_end_y),
                     border_color, 1, cv2.LINE_AA)


def draw_front_guides(f, pts, angles):
    for hip_i,knee_i,ankle_i,prefix in [(HIP_L,KNEE_L,ANKLE_L,'l'),(HIP_R,KNEE_R,ANKLE_R,'r')]:
        if hip_i in pts and knee_i in pts and ankle_i in pts:
            cv2.line(f,pts[hip_i],pts[ankle_i],(50,50,50),1,cv2.LINE_AA)
            t=(pts[knee_i][1]-pts[hip_i][1])/max(pts[ankle_i][1]-pts[hip_i][1],1)
            ix=int(pts[hip_i][0]+t*(pts[ankle_i][0]-pts[hip_i][0]))
            if abs(pts[knee_i][0]-ix)>3:
                fppa = abs(angles.get(f'{prefix}_fppa', 0))
                c = GREEN if fppa<8 else(YELLOW if fppa<12 else RED)
                cv2.arrowedLine(f,(ix,pts[knee_i][1]),pts[knee_i],c,2,cv2.LINE_AA,tipLength=0.25)
    ms=angles.get('mid_s'); mh=angles.get('mid_h')
    if ms and mh and abs(mh[0]-ms[0])>3:
        cv2.line(f,(ms[0],ms[1]-15),(ms[0],ms[1]+250),(50,50,50),1,cv2.LINE_AA)
        hs=abs(angles.get('hip_shift',0))
        sc=GREEN if hs<3 else(YELLOW if hs<8 else RED)
        cv2.arrowedLine(f,(ms[0],mh[1]),mh,sc,2,cv2.LINE_AA,tipLength=0.3)


# ── PANELS ──

def _draw_panel_rows(f, rows, px, py, pw):
    """Generic panel row renderer. Returns panel height."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    rh_ = 34
    header_count = sum(1 for r in rows if r[0] == 'header')
    spacer_count = sum(1 for r in rows if r[0] == 'spacer')
    data_count = sum(1 for r in rows if r[0] == 'data')
    panel_h = header_count * 38 + data_count * rh_ + spacer_count * 12 + 20

    ov = f.copy()
    cv2.rectangle(ov, (px-5, py-5), (px+pw+5, py+panel_h), DARK_BG, -1)
    cv2.addWeighted(ov, 0.78, f, 0.22, 0, f)
    cv2.line(f, (px, py), (px+pw, py), ACCENT, 2, cv2.LINE_AA)

    y = py + 8
    for row in rows:
        if row[0] == 'header':
            y += 28
            cv2.putText(f, row[1], (px+8, y), font, 0.58, ACCENT, 2, cv2.LINE_AA)
            y += 10
        elif row[0] == 'spacer':
            y += 12
        elif row[0] == 'data':
            _, label, val, color = row
            y += rh_
            cv2.circle(f, (px+10, y-4), 5, color, -1, cv2.LINE_AA)
            cv2.putText(f, label, (px+22, y), font, 0.45, (160,160,160), 1, cv2.LINE_AA)
            cv2.putText(f, val, (px+160, y), font, 0.48, color, 2, cv2.LINE_AA)

    return panel_h


def draw_frontal_panel(f, angles, rom_tracker, gait_tracker, w):
    """GAIT: FRONTAL PLANE panel."""
    pw = 400
    px = w - pw - 10
    py = 10
    rows = []

    rows.append(('header', 'GAIT: FRONTAL PLANE', None))

    # Hip ABD/ADD
    l_abd = angles.get('l_hip_abd', 0)
    r_abd = angles.get('r_hip_abd', 0)
    abd_asym = abs(abs(l_abd) - abs(r_abd))
    abd_color = RED if abd_asym > 10 else (YELLOW if abd_asym > 5 else GREEN)
    def abd_label(v):
        if v > 3: return f"ABD {v:.0f}\u00b0"
        elif v < -3: return f"ADD {abs(v):.0f}\u00b0"
        else: return "NEUT"
    rows.append(('data', 'HIP ABD', f"L:{abd_label(l_abd)}  R:{abd_label(r_abd)}", abd_color))

    # Knee Valgus/Varus (FPPA)
    lf = angles.get('l_fppa', 0)
    rf = angles.get('r_fppa', 0)
    valg_color = RED if max(abs(lf),abs(rf))>12 else(YELLOW if max(abs(lf),abs(rf))>8 else GREEN)
    rows.append(('data', 'VALGUS', f"L:{lf:+.0f}\u00b0 R:{rf:+.0f}\u00b0", valg_color))

    # Circumduction
    l_circ = gait_tracker.circumduction_index('l')
    r_circ = gait_tracker.circumduction_index('r')
    circ_thresh = 40
    worse_circ = max(l_circ, r_circ)
    circ_color = RED if worse_circ > circ_thresh*1.5 else (YELLOW if worse_circ > circ_thresh else GREEN)
    rows.append(('data', 'CIRCUMDUCT', f"L:{l_circ:.0f}px  R:{r_circ:.0f}px", circ_color))

    # Ankle eversion/inversion
    l_ev = angles.get('l_ev_inv')
    r_ev = angles.get('r_ev_inv')
    if l_ev is not None and r_ev is not None:
        def ev_label(v):
            if v > 3: return f"EV {v:.0f}\u00b0"
            elif v < -3: return f"INV {abs(v):.0f}\u00b0"
            else: return "NEUT"
        ev_asym = abs(l_ev - r_ev)
        ev_color = RED if ev_asym > 10 else (YELLOW if ev_asym > 5 else GREEN)
        rows.append(('data', 'ANKLE E/I', f"L:{ev_label(l_ev)}  R:{ev_label(r_ev)}", ev_color))

    rows.append(('spacer', '', None))

    # Pelvic obliquity
    po = angles.get('pelv_obliq', 0)
    po_color = RED if abs(po) > 15 else (YELLOW if abs(po) > 8 else GREEN)
    side = 'L low' if po > 0 else 'R low'
    rows.append(('data', 'PELVIS', f"{side} {abs(po):.0f}px", po_color))

    # Hip hiking
    l_hike, r_hike, obliq = gait_tracker.hip_hiking()
    obliq_color = RED if obliq > 20 else (YELLOW if obliq > 10 else GREEN)
    rows.append(('data', 'HIP HIKE', f"L:{l_hike:.0f}px  R:{r_hike:.0f}px", obliq_color))

    # Shoulder ABD
    l_sh = angles.get('l_sh_abd', 0)
    r_sh = angles.get('r_sh_abd', 0)
    sh_asym = abs(l_sh - r_sh)
    sh_color = RED if sh_asym > 15 else (YELLOW if sh_asym > 8 else GREEN)
    rows.append(('data', 'SH ABD', f"L:{l_sh:.0f}\u00b0  R:{r_sh:.0f}\u00b0", sh_color))

    rows.append(('spacer', '', None))

    # Arm swing ratio
    arm_ratio = gait_tracker.arm_swing_ratio()
    arm_color = RED if abs(arm_ratio - 1.0) > 0.8 else (YELLOW if abs(arm_ratio - 1.0) > 0.4 else GREEN)
    if arm_ratio > 1.0:
        arm_label = f"L/R {arm_ratio:.1f}x"
    else:
        arm_label = f"R/L {1.0/max(arm_ratio, 0.01):.1f}x"
    rows.append(('data', 'ARM SWING', arm_label, arm_color))

    # Step width (from 3D world coords)
    sw = gait_tracker.avg_step_width()
    # Normal step width ~0.08-0.12m
    sw_color = RED if sw > 0.20 else (YELLOW if sw > 0.15 else GREEN)
    if sw > 0:
        rows.append(('data', 'STEP WIDTH', f"{sw*100:.0f}cm", sw_color))

    # Shoulder tilt
    st = angles.get('sh_tilt', 0)
    st_color = YELLOW if st > 4 else GREEN
    rows.append(('data', 'SH TILT', f"{st:.0f}\u00b0", st_color))

    _draw_panel_rows(f, rows, px, py, pw)


def draw_sagittal_panel(f, angles, rom_tracker, gait_tracker, w):
    """GAIT: SAGITTAL PLANE panel."""
    pw = 400
    px = w - pw - 10
    py = 10
    rows = []

    rows.append(('header', 'GAIT: SAGITTAL PLANE', None))

    # Knee FLEX/EXT (show as 180-angle for flexion degrees)
    for side_label, key in [('L', 'l_knee'), ('R', 'r_knee')]:
        rom = rom_tracker.get_rom(key)
        if key in angles:
            cur = 180 - angles[key]
            if rom:
                mn, mx, _ = rom
                flex_rom = 180 - mn
                color = norm_color(flex_rom, 50, 135)
                rows.append(('data', f'KNEE {side_label}', f"FLEX {cur:.0f}\u00b0 (ROM {flex_rom:.0f}\u00b0)", color))
            else:
                rows.append(('data', f'KNEE {side_label}', f"FLEX {cur:.0f}\u00b0", GREEN))

    # Hip FLEX/EXT
    for side_label, key in [('L', 'l_hip'), ('R', 'r_hip')]:
        rom = rom_tracker.get_rom(key)
        if key in angles:
            cur = 180 - angles[key]
            if rom:
                mn, mx, _ = rom
                flex_rom = 180 - mn
                color = norm_color(flex_rom, 20, 120)
                rows.append(('data', f'HIP {side_label}', f"FLEX {cur:.0f}\u00b0 (ROM {flex_rom:.0f}\u00b0)", color))
            else:
                rows.append(('data', f'HIP {side_label}', f"FLEX {cur:.0f}\u00b0", GREEN))

    # Ankle DF/PF
    l_dfpf = angles.get('l_ankle_dfpf')
    r_dfpf = angles.get('r_ankle_dfpf')
    if l_dfpf is not None and r_dfpf is not None:
        l_label = get_ankle_label(l_dfpf)
        r_label = get_ankle_label(r_dfpf)
        asym = abs(l_dfpf - r_dfpf)
        color = RED if asym > 15 else (YELLOW if asym > 8 else GREEN)
        rows.append(('data', 'ANKLE', f"L:{l_label}  R:{r_label}", color))

    rows.append(('spacer', '', None))

    # Shoulder FLEX/EXT — already goniometric (0° = arm at side)
    for side_label, key in [('L', 'l_shoulder'), ('R', 'r_shoulder')]:
        if key in angles:
            cur = angles[key]
            if cur < 5:
                label = "NEUT"
            else:
                label = f"FLEX {cur:.0f}\u00b0"
            color = GREEN
            rows.append(('data', f'SHLDR {side_label}', label, color))

    # Trunk lean
    trunk = angles.get('trunk', 0)
    trunk_color = RED if trunk > 12 else (YELLOW if trunk > 8 else GREEN)
    rows.append(('data', 'TRUNK LEAN', f"{trunk:.0f}\u00b0", trunk_color))

    # CVA
    cva = angles.get('cva')
    if cva is not None:
        cva_color = norm_color(cva, 48, 90)
        rows.append(('data', 'CVA', f"{cva:.0f}\u00b0", cva_color))

    rows.append(('spacer', '', None))

    # Arm swing (still useful in sagittal)
    arm_ratio = gait_tracker.arm_swing_ratio()
    arm_color = RED if abs(arm_ratio - 1.0) > 0.8 else (YELLOW if abs(arm_ratio - 1.0) > 0.4 else GREEN)
    if arm_ratio > 1.0:
        arm_label = f"L/R {arm_ratio:.1f}x"
    else:
        arm_label = f"R/L {1.0/max(arm_ratio, 0.01):.1f}x"
    rows.append(('data', 'ARM SWING', arm_label, arm_color))

    _draw_panel_rows(f, rows, px, py, pw)


def draw_waveform(f, histories, w, h, wave_h=120):
    ns=len(histories[REGION_LABELS[0]])
    if ns<2: return
    yb=h-15; yt=yb-wave_h

    ov=f.copy()
    cv2.rectangle(ov,(0,yt-25),(w,h),(0,0,0),-1)
    cv2.addWeighted(ov,0.5,f,0.5,0,f)

    smoothed={r:smooth_wave(list(histories[r]),11) for r in REGION_LABELS}
    gmax=max(max(smoothed[r]) if smoothed[r] else 0 for r in REGION_LABELS)
    if gmax<=0: gmax=1.0
    n=len(smoothed[REGION_LABELS[0]]); step=w/max(n-1,1)
    curr={r:smoothed[r][-1] if smoothed[r] else 0 for r in REGION_LABELS}
    order=sorted(REGION_LABELS,key=lambda r:curr[r])

    for region in order:
        color=REGION_COLORS[region]; vals=smoothed[region]
        pts_list=[]
        for i,v in enumerate(vals):
            x=int(i*step); y=int(yb-(v/gmax)*wave_h); y=max(y,yt)
            pts_list.append([x,y])
        if len(pts_list)<2: continue
        lp=np.array(pts_list,dtype=np.int32).reshape((-1,1,2))
        dim=tuple(max(c//3,0) for c in color)
        cv2.polylines(f,[lp],False,dim,4,cv2.LINE_AA)
        bright=tuple(min(c+60,255) for c in color)
        thick=3 if region==order[-1] else 2
        cv2.polylines(f,[lp],False,bright,thick,cv2.LINE_AA)

    cx=int((n-1)*step)
    cv2.line(f,(cx,yt-25),(cx,yb),WHITE,1,cv2.LINE_AA)
    cv2.line(f,(0,yb),(w,yb),(60,60,60),1,cv2.LINE_AA)

    ly=yt-10; lx=10
    for r in REGION_LABELS:
        bright=tuple(min(c+60,255) for c in REGION_COLORS[r])
        cv2.rectangle(f,(lx,ly-6),(lx+8,ly+2),bright,-1)
        cv2.putText(f,REGION_DISPLAY[r],(lx+11,ly+1),cv2.FONT_HERSHEY_SIMPLEX,0.45,WHITE,1,cv2.LINE_AA)
        lx+=11+len(REGION_DISPLAY[r])*10+14


# ── MAIN ──
def process(input_path, output_path):
    cap=cv2.VideoCapture(input_path)
    if not cap.isOpened(): print(f"ERROR: Cannot open {input_path}"); return

    w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps=cap.get(cv2.CAP_PROP_FPS) or 30.0
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing: {input_path} \u2014 {w}x{h} @ {fps:.1f}fps, {total} frames")

    out=cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*'avc1'),fps,(w,h))
    options=PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='/tmp/flekks-viz/pose_landmarker_heavy.task'),
        running_mode=RunningMode.VIDEO,num_poses=1,
        min_pose_detection_confidence=0.5,min_tracking_confidence=0.6)
    landmarker=PoseLandmarker.create_from_options(options)

    smoother=Smoother(alpha=SMOOTH_ALPHA)
    world_smoother=WorldSmoother(alpha=SMOOTH_ALPHA)
    rom_tracker = ROMTracker(window=int(fps*3))
    gait_tracker = GaitTracker(window=int(fps*3))
    prev_pts={}
    histories={r:deque(maxlen=int(fps*8)) for r in REGION_LABELS}
    view_votes={'front':0,'side':0}
    stable_view='front'

    fidx=0; ts=0; dt=1000.0/fps

    while True:
        ret,frame=cap.read()
        if not ret: break

        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        mp_img=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
        result=landmarker.detect_for_video(mp_img,int(ts)); ts+=dt

        # 2D landmarks for drawing
        lm=result.pose_landmarks[0] if result.pose_landmarks and len(result.pose_landmarks)>0 else None
        pts=smoother.update(lm,w,h)

        # 3D world landmarks for angle calculations
        world_lm=result.pose_world_landmarks[0] if result.pose_world_landmarks and len(result.pose_world_landmarks)>0 else None
        world_pts=world_smoother.update(world_lm)

        mv=compute_movement(prev_pts,pts)
        for r in REGION_LABELS:
            histories[r].append(mv[r])
        prev_pts=pts

        v=detect_view(pts, w)
        if fidx<30: view_votes[v]+=1
        elif fidx==30:
            stable_view='front' if view_votes['front']>=view_votes['side'] else 'side'
            print(f"  Detected view: {stable_view}")
        if fidx>=30: v=stable_view

        # Compute angles from 3D world landmarks, with 2D pts for overlay metrics
        angles = compute_joint_angles(world_pts, pts)
        rom_tracker.update(angles)
        gait_tracker.update(pts, world_pts)

        draw_skeleton(frame, pts)
        draw_joint_angles(frame, pts, angles, v)
        if v=='front':
            draw_front_guides(frame, pts, angles)
            draw_frontal_panel(frame, angles, rom_tracker, gait_tracker, w)
        else:
            draw_sagittal_panel(frame, angles, rom_tracker, gait_tracker, w)
        draw_waveform(frame, histories, w, h)

        out.write(frame)
        fidx+=1
        if fidx%50==0: print(f"  {fidx}/{total} ({100*fidx//total}%)")

    cap.release(); out.release(); landmarker.close()
    print(f"Done: {output_path} ({fidx} frames)")


if __name__=="__main__":
    if len(sys.argv)>=3: process(sys.argv[1],sys.argv[2])
    else: process("/tmp/flekks-viz/input.mp4","/tmp/flekks-viz/gait_output.mp4")
