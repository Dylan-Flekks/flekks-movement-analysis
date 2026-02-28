#!/usr/bin/env python3
"""
Flekks PT Analyzer v6
- AutoCAD-style arc indicators at each joint showing angle origin
- Flexion/Extension labels (not just raw degrees)
- Polished visual styling with gradient backgrounds & glow effects
- Data-driven natural language assessment + diagnostic recommendations
- Intensity waveform at bottom
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import sys
import textwrap
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
JOINTS = {7,8,11,12,13,14,15,16,23,24,25,26,27,28,29,30,31,32}
SMOOTH_ALPHA = 0.45

# Joint classification: what the angle means at each joint
# 180° = fully extended (straight), less = flexion
JOINT_TYPES = {
    'l_elbow': ('Elbow', 'FLEX', 'EXT'),      # <180 = flexion
    'r_elbow': ('Elbow', 'FLEX', 'EXT'),
    'l_shoulder': ('Shoulder', 'FLEX', 'EXT'),
    'r_shoulder': ('Shoulder', 'FLEX', 'EXT'),
    'l_hip': ('Hip', 'FLEX', 'EXT'),
    'r_hip': ('Hip', 'FLEX', 'EXT'),
    'l_knee': ('Knee', 'FLEX', 'EXT'),         # <180 = flexion
    'r_knee': ('Knee', 'FLEX', 'EXT'),
    'l_ankle': ('Ankle', 'DF', 'PF'),          # dorsiflexion / plantarflexion
    'r_ankle': ('Ankle', 'DF', 'PF'),
}

# Clinical norms
NORMS = {
    'knee_flexion': (0, 135),
    'hip_flexion': (0, 120),
    'elbow_flexion': (0, 145),
    'shoulder_flexion': (0, 180),
    'knee_valgus': (-8, 8),
    'ankle_df': (10, 25),
    'cva': (48, 90),
    'trunk_lean': (0, 8),
    'pelvic_tilt': (7, 15),
    'kyphosis': (20, 40),
    'lordosis': (30, 50),
}

# ── Joint landmark triplets for arc drawing (parent, vertex, child) ──
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


class Smoother:
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


def mid(a, b): return ((a[0]+b[0])//2,(a[1]+b[1])//2)


def angle_at(p1, v, p2):
    v1=np.array([p1[0]-v[0],p1[1]-v[1]],dtype=np.float64)
    v2=np.array([p2[0]-v[0],p2[1]-v[1]],dtype=np.float64)
    n1,n2=np.linalg.norm(v1),np.linalg.norm(v2)
    if n1<1 or n2<1: return 180.0
    c=np.dot(v1,v2)/(n1*n2)
    return np.degrees(np.arccos(np.clip(c,-1,1)))


def detect_view(pts):
    if SHOULDER_L not in pts or SHOULDER_R not in pts: return 'front'
    dx=abs(pts[SHOULDER_L][0]-pts[SHOULDER_R][0])
    hw=abs(pts[HIP_L][0]-pts[HIP_R][0]) if HIP_L in pts and HIP_R in pts else 0
    return 'side' if (dx+hw)/2 < 80 else 'front'


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


# ── COMPUTE ALL JOINT ANGLES ──
def compute_joint_angles(pts):
    a = {}

    # Left side
    if SHOULDER_L in pts and ELBOW_L in pts and WRIST_L in pts:
        a['l_elbow'] = angle_at(pts[SHOULDER_L], pts[ELBOW_L], pts[WRIST_L])
    if HIP_L in pts and SHOULDER_L in pts and ELBOW_L in pts:
        a['l_shoulder'] = angle_at(pts[HIP_L], pts[SHOULDER_L], pts[ELBOW_L])
    if SHOULDER_L in pts and HIP_L in pts and KNEE_L in pts:
        a['l_hip'] = angle_at(pts[SHOULDER_L], pts[HIP_L], pts[KNEE_L])
    if HIP_L in pts and KNEE_L in pts and ANKLE_L in pts:
        a['l_knee'] = angle_at(pts[HIP_L], pts[KNEE_L], pts[ANKLE_L])
    if KNEE_L in pts and ANKLE_L in pts and FOOT_L in pts:
        a['l_ankle'] = angle_at(pts[KNEE_L], pts[ANKLE_L], pts[FOOT_L])

    # Right side
    if SHOULDER_R in pts and ELBOW_R in pts and WRIST_R in pts:
        a['r_elbow'] = angle_at(pts[SHOULDER_R], pts[ELBOW_R], pts[WRIST_R])
    if HIP_R in pts and SHOULDER_R in pts and ELBOW_R in pts:
        a['r_shoulder'] = angle_at(pts[HIP_R], pts[SHOULDER_R], pts[ELBOW_R])
    if SHOULDER_R in pts and HIP_R in pts and KNEE_R in pts:
        a['r_hip'] = angle_at(pts[SHOULDER_R], pts[HIP_R], pts[KNEE_R])
    if HIP_R in pts and KNEE_R in pts and ANKLE_R in pts:
        a['r_knee'] = angle_at(pts[HIP_R], pts[KNEE_R], pts[ANKLE_R])
    if KNEE_R in pts and ANKLE_R in pts and FOOT_R in pts:
        a['r_ankle'] = angle_at(pts[KNEE_R], pts[ANKLE_R], pts[FOOT_R])

    # Trunk angle
    if SHOULDER_L in pts and HIP_L in pts:
        s,h=pts[SHOULDER_L],pts[HIP_L]
        a['trunk'] = np.degrees(np.arctan2(abs(s[0]-h[0]), max(abs(h[1]-s[1]),1)))

    # Asymmetries
    for joint in ['elbow','shoulder','hip','knee','ankle']:
        lk, rk = f'l_{joint}', f'r_{joint}'
        if lk in a and rk in a:
            a[f'{joint}_asym'] = abs(a[lk] - a[rk])

    # FPPA (knee valgus/varus)
    for hip_i,knee_i,ankle_i,prefix in [(HIP_L,KNEE_L,ANKLE_L,'l'),(HIP_R,KNEE_R,ANKLE_R,'r')]:
        if hip_i in pts and knee_i in pts and ankle_i in pts:
            v1=np.array([pts[hip_i][0]-pts[knee_i][0],pts[hip_i][1]-pts[knee_i][1]],dtype=np.float64)
            v2=np.array([pts[ankle_i][0]-pts[knee_i][0],pts[ankle_i][1]-pts[knee_i][1]],dtype=np.float64)
            n1,n2=np.linalg.norm(v1),np.linalg.norm(v2)
            if n1>1 and n2>1:
                c=np.dot(v1,v2)/(n1*n2)
                dev=180.0-np.degrees(np.arccos(np.clip(c,-1,1)))
                cross=v1[0]*v2[1]-v1[1]*v2[0]
                if prefix=='l':
                    a[f'{prefix}_fppa']=dev if cross>0 else -dev
                else:
                    a[f'{prefix}_fppa']=-dev if cross<0 else dev

    # Hip shift
    ms=mid(pts[SHOULDER_L],pts[SHOULDER_R]) if SHOULDER_L in pts and SHOULDER_R in pts else None
    mh=mid(pts[HIP_L],pts[HIP_R]) if HIP_L in pts and HIP_R in pts else None
    if ms and mh:
        sw=max(abs(pts[SHOULDER_R][0]-pts[SHOULDER_L][0]),1)
        a['hip_shift']=(mh[0]-ms[0])/sw*100
    a['mid_s']=ms; a['mid_h']=mh

    # Shoulder tilt
    if SHOULDER_L in pts and SHOULDER_R in pts:
        dy=pts[SHOULDER_R][1]-pts[SHOULDER_L][1]
        dx=max(abs(pts[SHOULDER_R][0]-pts[SHOULDER_L][0]),1)
        a['sh_tilt']=np.degrees(np.arctan2(abs(dy),dx))

    # Pelvic drop
    if HIP_L in pts and HIP_R in pts:
        dy=abs(pts[HIP_L][1]-pts[HIP_R][1])
        ref=abs(pts[SHOULDER_L][1]-pts[HIP_L][1]) if SHOULDER_L in pts else 100
        a['pelv_drop']=(dy/max(ref,1))*100

    # CVA (side view)
    for ear_i,sh_i in [(EAR_L,SHOULDER_L),(EAR_R,SHOULDER_R)]:
        if ear_i in pts and sh_i in pts:
            e,s=pts[ear_i],pts[sh_i]
            a['cva']=np.degrees(np.arctan2(s[1]-e[1],abs(e[0]-s[0])))
            break

    # Kyphosis estimate
    ear_i = EAR_L if EAR_L in pts else EAR_R
    sh_i = SHOULDER_L if SHOULDER_L in pts else SHOULDER_R
    hip_i = HIP_L if HIP_L in pts else HIP_R
    if ear_i in pts and sh_i in pts and hip_i in pts:
        a['kyphosis'] = max(0, 180.0 - angle_at(pts[ear_i], pts[sh_i], pts[hip_i]))

    # Lordosis estimate
    knee_i = KNEE_L if KNEE_L in pts else KNEE_R
    if sh_i in pts and hip_i in pts and knee_i in pts:
        a['lordosis'] = max(0, 180.0 - angle_at(pts[sh_i], pts[hip_i], pts[knee_i]))

    return a


# ── DATA-DRIVEN NATURAL LANGUAGE ──
def generate_assessment(view, angles, intensity):
    lines = []
    recs = []

    if view == 'front':
        l_fp = angles.get('l_fppa', 0)
        r_fp = angles.get('r_fppa', 0)

        if abs(l_fp) > 12 or abs(r_fp) > 12:
            worse = 'left' if abs(l_fp) > abs(r_fp) else 'right'
            val = max(abs(l_fp), abs(r_fp))
            ktype = 'valgus (inward)' if (l_fp > 0 if worse=='left' else r_fp > 0) else 'varus (outward)'
            lines.append(f"Your {worse} knee is moving {ktype} at {val:.0f}°. Glute medius likely not firing enough.")
            recs.append(f"NEXT: Film from the side to check if tight ankles cause knee compensation.")
        elif abs(l_fp) > 8 or abs(r_fp) > 8:
            worse = 'left' if abs(l_fp) > abs(r_fp) else 'right'
            val = max(abs(l_fp), abs(r_fp))
            ktype = 'inward' if (l_fp > 0 if worse=='left' else r_fp > 0) else 'outward'
            lines.append(f"Mild {worse} knee drift {ktype} ({val:.0f}°). Worth strengthening hip muscles.")

        ka = angles.get('knee_asym', 0)
        if ka > 8:
            lk = angles.get('l_knee', 180)
            rk = angles.get('r_knee', 180)
            tighter = 'left' if lk > rk else 'right'
            lines.append(f"Knees bending unevenly — {ka:.0f}° diff. {tighter.title()} side stiffer.")
            recs.append(f"NEXT: Single-leg squat to isolate which side is limited.")

        hs = angles.get('hip_shift', 0)
        if abs(hs) > 6:
            side = 'right' if hs > 0 else 'left'
            lines.append(f"Hips shift {side} by {abs(hs):.0f}%. Possible glute weakness on opposite side.")
            recs.append(f"NEXT: Single-leg bridge test to confirm glute weakness.")

        ha = angles.get('hip_asym', 0)
        if ha > 10:
            lh = angles.get('l_hip', 180)
            rh = angles.get('r_hip', 180)
            lines.append(f"Hip ROM asymmetric — L: {lh:.0f}° vs R: {rh:.0f}° ({ha:.0f}° diff).")

        st = angles.get('sh_tilt', 0)
        if st > 4:
            lines.append(f"Shoulders uneven ({st:.0f}° tilt). Possible upper trap or lat imbalance.")

        l_int = intensity.get('left_leg', 0)
        r_int = intensity.get('right_leg', 0)
        if l_int > 0 and r_int > 0:
            ratio = max(l_int, r_int) / max(min(l_int, r_int), 0.1)
            if ratio > 1.8:
                dom = 'left' if l_int > r_int else 'right'
                lines.append(f"{dom.title()} leg doing more work ({ratio:.1f}x). May be favoring that side.")

    elif view == 'side':
        cva = angles.get('cva', 50)
        if cva < 45:
            lines.append(f"Head {50-cva:.0f}° forward (CVA: {cva:.0f}°). Extra neck load. Deep neck flexors likely weak.")
            recs.append(f"NEXT: Film side-sitting to compare posture.")
        elif cva < 50:
            lines.append(f"Slight forward head (CVA: {cva:.0f}°). Watch screen posture.")

        kyph = angles.get('kyphosis', 30)
        if kyph > 45:
            lines.append(f"Upper back rounding at {kyph:.0f}° (normal 20-40°). Tight pecs, weak mid-back.")
            recs.append(f"NEXT: Film from front with arms overhead to check shoulder reach.")
        elif kyph < 18:
            lines.append(f"Very flat upper back ({kyph:.0f}°). Reduced shock absorption.")

        lord = angles.get('lordosis', 40)
        if lord > 55:
            lines.append(f"Excessive low back curve ({lord:.0f}°, normal 30-50°). Tight hip flexors, weak core.")
            recs.append(f"NEXT: Thomas test — pull knee to chest, check if other leg lifts.")
        elif lord < 25:
            lines.append(f"Flat low back ({lord:.0f}°). Tight hamstrings pulling pelvis under.")
            recs.append(f"NEXT: Toe-touch test for hamstring flexibility.")

        trunk = angles.get('trunk', 0)
        if trunk > 12:
            lines.append(f"Leaning forward {trunk:.0f}°. Loads low back. Tight ankles or weak glutes.")

        lk = angles.get('l_knee', angles.get('r_knee', 180))
        if lk > 188:
            lines.append(f"Knee hyperextending ({lk:.0f}°). Hamstrings not controlling end range.")
        elif lk < 168:
            lines.append(f"Knee staying bent at {lk:.0f}°. Tight hamstrings or quad weakness.")

        la = max(intensity.get('left_arm', 0), intensity.get('right_arm', 0))
        ll = max(intensity.get('left_leg', 0), intensity.get('right_leg', 0))
        if la > 0 and ll > 0 and la/max(ll,0.1) > 3:
            lines.append(f"Arms compensating — {la/ll:.1f}x more movement than legs. Trunk may be unstable.")

    if not lines:
        lines.append("Movement looks good. No significant issues from this angle.")
        recs.append("NEXT: Film from opposite angle for complete picture.")

    if not recs:
        recs.append("NEXT: Film from opposite angle to check for hidden issues.")

    return lines, recs


# ── DRAWING ──

def draw_skeleton(f, pts):
    """Draw skeleton with glow effect."""
    # Outer glow
    for a,b in SKELETON:
        if a in pts and b in pts:
            cv2.line(f,pts[a],pts[b],ACCENT_DIM,6,cv2.LINE_AA)
    # Core line
    for a,b in SKELETON:
        if a in pts and b in pts:
            cv2.line(f,pts[a],pts[b],ACCENT,2,cv2.LINE_AA)
    # Joints with glow
    for i in JOINTS:
        if i in pts:
            cv2.circle(f,pts[i],6,ACCENT_DIM,-1,cv2.LINE_AA)
            cv2.circle(f,pts[i],4,ACCENT,-1,cv2.LINE_AA)
            cv2.circle(f,pts[i],2,WHITE,-1,cv2.LINE_AA)


def draw_arc(f, vertex, p1, p2, angle_deg, color, radius=28, thickness=2):
    """Draw an AutoCAD-style arc between two limb segments at a joint.

    vertex: the joint point
    p1: parent bone end
    p2: child bone end
    angle_deg: the measured angle
    color: arc color
    """
    vx, vy = vertex
    # Vectors from vertex to each point
    dx1, dy1 = float(p1[0]-vx), float(p1[1]-vy)
    dx2, dy2 = float(p2[0]-vx), float(p2[1]-vy)

    n1 = math.sqrt(dx1*dx1 + dy1*dy1)
    n2 = math.sqrt(dx2*dx2 + dy2*dy2)
    if n1 < 1 or n2 < 1:
        return

    # Angles from vertex to each bone direction (atan2 in degrees)
    ang1 = math.degrees(math.atan2(-dy1, dx1))  # negative y because image coords
    ang2 = math.degrees(math.atan2(-dy2, dx2))

    # Determine start/end for the arc (sweep the smaller angle)
    diff = (ang2 - ang1) % 360
    if diff > 180:
        start_angle = ang2
        sweep = 360 - diff
    else:
        start_angle = ang1
        sweep = diff

    # Draw the arc using ellipse
    cv2.ellipse(f, (vx, vy), (radius, radius), 0,
                -start_angle - sweep, -start_angle,  # OpenCV uses clockwise angles
                color, thickness, cv2.LINE_AA)

    # Small tick marks at arc ends (the "dimension line" feel)
    for ang in [ang1, ang2]:
        rad = math.radians(ang)
        ex = int(vx + (radius - 4) * math.cos(rad))
        ey = int(vy - (radius - 4) * math.sin(rad))
        ex2 = int(vx + (radius + 4) * math.cos(rad))
        ey2 = int(vy - (radius + 4) * math.sin(rad))
        cv2.line(f, (ex, ey), (ex2, ey2), color, 1, cv2.LINE_AA)


def get_flexion_label(key, angle_deg):
    """Return human-readable label: 'FLEX 85°' or 'EXT 175°' etc."""
    if key not in JOINT_TYPES:
        return f"{angle_deg:.0f}°"

    joint_name, flex_label, ext_label = JOINT_TYPES[key]

    # Convention: 180° = fully straight/extended
    # Less than ~170° = some degree of flexion
    # For ankle: angle < 90° = dorsiflexion, > 90° = plantarflexion
    if key.endswith('ankle'):
        if angle_deg < 85:
            return f"DF {90-angle_deg:.0f}°"
        elif angle_deg > 95:
            return f"PF {angle_deg-90:.0f}°"
        else:
            return f"NEUT"

    if angle_deg >= 175:
        return f"{ext_label}"
    elif angle_deg >= 160:
        return f"{flex_label} {180-angle_deg:.0f}°"
    else:
        return f"{flex_label} {180-angle_deg:.0f}°"


def draw_joint_angles(f, pts, angles):
    """Draw AutoCAD-style arcs + flexion/extension labels at each joint."""

    for key, (p1_i, vertex_i, p2_i) in JOINT_ARCS.items():
        if key not in angles:
            continue
        if p1_i not in pts or vertex_i not in pts or p2_i not in pts:
            continue

        val = angles[key]
        vertex = pts[vertex_i]
        p1 = pts[p1_i]
        p2 = pts[p2_i]

        # Color based on L/R asymmetry
        joint_base = key[2:]  # e.g. 'knee'
        asym = angles.get(f'{joint_base}_asym', 0)

        if asym > 12:
            color = RED
            label_color = RED
        elif asym > 6:
            color = ORANGE
            label_color = YELLOW
        else:
            color = CYAN
            label_color = CYAN

        # Don't draw arcs for nearly straight joints (>170°) — too noisy
        if val > 170:
            arc_color = tuple(max(c//2, 0) for c in color)  # dim for near-extension
            radius = 20
        else:
            arc_color = color
            radius = min(30, max(18, int(40 * (180 - val) / 180)))

        # Draw the arc
        draw_arc(f, vertex, p1, p2, val, arc_color, radius=radius, thickness=2)

        # Label placement: offset away from the body center
        # Use the bisector of the two vectors to place the label outward
        dx1 = float(p1[0]-vertex[0])
        dy1 = float(p1[1]-vertex[1])
        dx2 = float(p2[0]-vertex[0])
        dy2 = float(p2[1]-vertex[1])
        n1 = max(math.sqrt(dx1*dx1+dy1*dy1), 1)
        n2 = max(math.sqrt(dx2*dx2+dy2*dy2), 1)

        # Bisector direction (outward from the angle)
        bx = dx1/n1 + dx2/n2
        by = dy1/n1 + dy2/n2
        bn = max(math.sqrt(bx*bx+by*by), 0.01)
        bx /= bn
        by /= bn

        label = get_flexion_label(key, val)
        label_offset = radius + 14

        tx = int(vertex[0] + bx * label_offset)
        ty = int(vertex[1] + by * label_offset)

        # Clamp to frame
        tx = max(5, min(tx, f.shape[1]-80))
        ty = max(12, min(ty, f.shape[0]-5))

        # Styled label background with rounded feel
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.33
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)

        # Background pill shape
        pad_x, pad_y = 5, 3
        x1, y1 = tx - pad_x, ty - th - pad_y
        x2, y2 = tx + tw + pad_x, ty + pad_y + 1

        # Semi-transparent dark background
        overlay = f.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), DARK_BG, -1)
        cv2.addWeighted(overlay, 0.8, f, 0.2, 0, f)

        # Thin border matching the color
        border_color = tuple(max(c//2, 0) for c in label_color)
        cv2.rectangle(f, (x1, y1), (x2, y2), border_color, 1, cv2.LINE_AA)

        # Label text
        cv2.putText(f, label, (tx, ty), font, font_scale, label_color, 1, cv2.LINE_AA)

        # Small leader line from arc to label
        arc_edge_x = int(vertex[0] + bx * radius)
        arc_edge_y = int(vertex[1] + by * radius)
        label_near_x = int(vertex[0] + bx * (label_offset - 6))
        label_near_y = int(vertex[1] + by * (label_offset - 6))
        cv2.line(f, (arc_edge_x, arc_edge_y), (label_near_x, label_near_y),
                 border_color, 1, cv2.LINE_AA)

    # FPPA on knees (front view) — styled
    for prefix, knee_i, ox in [('l', KNEE_L, -55), ('r', KNEE_R, 15)]:
        fppa_key = f'{prefix}_fppa'
        if fppa_key in angles and knee_i in pts:
            val = angles[fppa_key]
            pt = pts[knee_i]

            if abs(val) < 3:
                ktype = 'NEUTRAL'
            elif val > 0:
                ktype = f'VALGUS {abs(val):.0f}°'
            else:
                ktype = f'VARUS {abs(val):.0f}°'

            color = GREEN if abs(val) < 8 else (YELLOW if abs(val) < 12 else RED)

            tx, ty = pt[0]+ox, pt[1]+22
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.28
            (tw, th), _ = cv2.getTextSize(ktype, font, font_scale, 1)

            # Pill background
            pad_x, pad_y = 4, 2
            overlay = f.copy()
            cv2.rectangle(overlay, (tx-pad_x, ty-th-pad_y), (tx+tw+pad_x, ty+pad_y+1), DARK_BG, -1)
            cv2.addWeighted(overlay, 0.8, f, 0.2, 0, f)
            border = tuple(max(c//2, 0) for c in color)
            cv2.rectangle(f, (tx-pad_x, ty-th-pad_y), (tx+tw+pad_x, ty+pad_y+1), border, 1, cv2.LINE_AA)
            cv2.putText(f, ktype, (tx, ty), font, font_scale, color, 1, cv2.LINE_AA)

    # Trunk angle between shoulders and hips
    if 'trunk' in angles and SHOULDER_L in pts:
        val = angles['trunk']
        color = GREEN if val < 5 else (YELLOW if val < 10 else RED)
        ms = mid(pts[SHOULDER_L], pts[SHOULDER_R]) if SHOULDER_R in pts else pts[SHOULDER_L]
        mh = mid(pts[HIP_L], pts[HIP_R]) if HIP_L in pts and HIP_R in pts else None
        if mh:
            mp_ = mid(ms, mh)
            label = f"TRUNK {val:.0f}°"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(label, font, 0.3, 1)
            tx, ty = mp_[0]-tw//2, mp_[1]
            overlay = f.copy()
            cv2.rectangle(overlay, (tx-4, ty-th-3), (tx+tw+4, ty+4), DARK_BG, -1)
            cv2.addWeighted(overlay, 0.75, f, 0.25, 0, f)
            cv2.putText(f, label, (tx, ty), font, 0.3, color, 1, cv2.LINE_AA)

    # CVA near ear (side view)
    if 'cva' in angles:
        for ear_i in [EAR_L, EAR_R]:
            if ear_i in pts:
                val = angles['cva']
                color = GREEN if val >= 50 else (YELLOW if val >= 45 else RED)
                label = f"CVA {val:.0f}°"
                font = cv2.FONT_HERSHEY_SIMPLEX
                (tw, th), _ = cv2.getTextSize(label, font, 0.32, 1)
                tx = pts[ear_i][0] - tw//2
                ty = pts[ear_i][1] - 18
                overlay = f.copy()
                cv2.rectangle(overlay, (tx-4, ty-th-3), (tx+tw+4, ty+4), DARK_BG, -1)
                cv2.addWeighted(overlay, 0.75, f, 0.25, 0, f)
                border = tuple(max(c//2, 0) for c in color)
                cv2.rectangle(f, (tx-4, ty-th-3), (tx+tw+4, ty+4), border, 1, cv2.LINE_AA)
                cv2.putText(f, label, (tx, ty), font, 0.32, color, 1, cv2.LINE_AA)
                break


def draw_front_guides(f, pts, angles):
    for hip_i,knee_i,ankle_i,prefix in [(HIP_L,KNEE_L,ANKLE_L,'l'),(HIP_R,KNEE_R,ANKLE_R,'r')]:
        if hip_i in pts and knee_i in pts and ankle_i in pts:
            # Dotted plumb line from hip to ankle
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


def draw_side_guides(f, pts):
    for ankle_i in [ANKLE_L, ANKLE_R]:
        if ankle_i in pts:
            top = 20
            for ear_i in [EAR_L, EAR_R]:
                if ear_i in pts: top = min(pts[ear_i][1]-30, top)
            # Dashed plumb line from ankle up
            ax = pts[ankle_i][0]
            y = top
            while y < pts[ankle_i][1]:
                y2 = min(y+8, pts[ankle_i][1])
                cv2.line(f,(ax,y),(ax,y2),(50,80,50),1,cv2.LINE_AA)
                y += 14
            break


def draw_metrics_panel(f, view, angles, w):
    """Top-right: measured data panel with polished styling."""
    pw = 240
    px = w - pw - 10
    py = 10

    if view == 'front':
        rows = []
        lk = angles.get('l_knee', None)
        rk = angles.get('r_knee', None)
        ka = angles.get('knee_asym', 0)
        if lk and rk:
            l_label = get_flexion_label('l_knee', lk)
            r_label = get_flexion_label('r_knee', rk)
            rows.append(('KNEE', f"L:{l_label}  R:{r_label}",
                         RED if ka>10 else(YELLOW if ka>5 else GREEN)))
            if ka > 3:
                rows.append(('  DIFF', f"{ka:.0f}°",
                             RED if ka>10 else(YELLOW if ka>5 else GREEN)))

        lf = angles.get('l_fppa', 0)
        rf = angles.get('r_fppa', 0)
        rows.append(('VALGUS', f"L:{lf:+.0f}° R:{rf:+.0f}°",
                     RED if max(abs(lf),abs(rf))>12 else(YELLOW if max(abs(lf),abs(rf))>8 else GREEN)))

        lh = angles.get('l_hip', None)
        rh = angles.get('r_hip', None)
        ha = angles.get('hip_asym', 0)
        if lh and rh:
            l_hl = get_flexion_label('l_hip', lh)
            r_hl = get_flexion_label('r_hip', rh)
            rows.append(('HIP', f"L:{l_hl}  R:{r_hl}",
                         RED if ha>10 else(YELLOW if ha>5 else GREEN)))

        hs = angles.get('hip_shift', 0)
        rows.append(('HIP SHIFT', f"{hs:+.0f}%",
                     RED if abs(hs)>8 else(YELLOW if abs(hs)>4 else GREEN)))

        st = angles.get('sh_tilt', 0)
        rows.append(('SHOULDER', f"Tilt {st:.0f}°",
                     YELLOW if st>3 else GREEN))

        # Elbow info if available
        le = angles.get('l_elbow', None)
        re = angles.get('r_elbow', None)
        if le and re:
            ea = angles.get('elbow_asym', 0)
            rows.append(('ELBOW', f"L:{le:.0f}° R:{re:.0f}°",
                         YELLOW if ea > 8 else CYAN))

    else:  # side
        rows = [
            ('HEAD', f"CVA {angles.get('cva',50):.0f}°",
             norm_color(angles.get('cva',50), 48, 90)),
            ('UPPER BACK', f"Kyphosis {angles.get('kyphosis',30):.0f}°",
             norm_color(angles.get('kyphosis',30), 20, 40)),
            ('LOW BACK', f"Lordosis {angles.get('lordosis',40):.0f}°",
             norm_color(angles.get('lordosis',40), 30, 50)),
            ('TRUNK', f"Lean {angles.get('trunk',0):.0f}°",
             norm_color(angles.get('trunk',0), 0, 8)),
        ]
        lk = angles.get('l_knee', angles.get('r_knee', None))
        if lk:
            label = get_flexion_label('l_knee', lk)
            rows.append(('KNEE', label,
                         GREEN if 165<=lk<=185 else YELLOW))
        le = angles.get('l_elbow', angles.get('r_elbow', None))
        if le:
            label = get_flexion_label('l_elbow', le)
            rows.append(('ELBOW', label, CYAN))
        lhi = angles.get('l_hip', angles.get('r_hip', None))
        if lhi:
            label = get_flexion_label('l_hip', lhi)
            rows.append(('HIP', label, CYAN))

    rh_ = 22
    panel_h = len(rows)*rh_ + 30

    # Semi-transparent panel background
    ov=f.copy()
    cv2.rectangle(ov,(px-5,py-5),(px+pw+5,py+panel_h),DARK_BG,-1)
    cv2.addWeighted(ov,0.78,f,0.22,0,f)

    # Accent line at top
    cv2.line(f,(px,py),(px+pw,py),ACCENT,2,cv2.LINE_AA)

    title = 'JOINT ANALYSIS' if view=='front' else 'POSTURE ANALYSIS'
    cv2.putText(f,title,(px+8,py+16),cv2.FONT_HERSHEY_SIMPLEX,0.42,ACCENT,1,cv2.LINE_AA)

    y=py+35
    for label,val,color in rows:
        # Status dot
        cv2.circle(f,(px+8,y-3),3,color,-1,cv2.LINE_AA)
        # Label
        cv2.putText(f,label,(px+16,y),cv2.FONT_HERSHEY_SIMPLEX,0.28,(160,160,160),1,cv2.LINE_AA)
        # Value (right-aligned feel)
        cv2.putText(f,val,(px+100,y),cv2.FONT_HERSHEY_SIMPLEX,0.30,color,1,cv2.LINE_AA)
        y+=rh_

    return py + panel_h


def draw_assessment(f, findings, recs, w, panel_top):
    """Plain English below metrics."""
    pw = 240
    px = w - pw - 10
    py = panel_top + 8

    wrapped = []
    for line in findings[:3]:
        wrapped.extend(textwrap.wrap(line, width=38))
        wrapped.append('')

    if recs:
        wrapped.append('---')
        for r in recs[:2]:
            wrapped.extend(textwrap.wrap(r, width=38))
            wrapped.append('')

    lh = 12
    panel_h = len(wrapped)*lh + 22

    ov=f.copy()
    cv2.rectangle(ov,(px-5,py-5),(px+pw+5,py+panel_h),DARK_BG,-1)
    cv2.addWeighted(ov,0.72,f,0.28,0,f)

    # Section header
    cv2.putText(f,"WHAT THIS MEANS",(px+8,py+10),cv2.FONT_HERSHEY_SIMPLEX,0.3,(150,150,150),1,cv2.LINE_AA)

    y=py+22
    for line in wrapped:
        if not line: y+=3; continue
        if line == '---':
            cv2.line(f,(px+8,y),(px+pw-8,y),(60,60,60),1)
            y+=6; continue
        if line.startswith('NEXT:'):
            color = ACCENT
        else:
            color = (195,195,195)
        cv2.putText(f,line,(px+8,y),cv2.FONT_HERSHEY_SIMPLEX,0.25,color,1,cv2.LINE_AA)
        y+=lh


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
        cv2.putText(f,REGION_DISPLAY[r],(lx+11,ly+1),cv2.FONT_HERSHEY_SIMPLEX,0.28,WHITE,1,cv2.LINE_AA)
        lx+=11+len(REGION_DISPLAY[r])*6+10


# ── MAIN ──
def process(input_path, output_path):
    cap=cv2.VideoCapture(input_path)
    if not cap.isOpened(): print(f"ERROR: Cannot open {input_path}"); return

    w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps=cap.get(cv2.CAP_PROP_FPS) or 30.0
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing: {input_path} — {w}x{h} @ {fps:.1f}fps, {total} frames")

    out=cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
    options=PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='/tmp/flekks-viz/pose_landmarker_heavy.task'),
        running_mode=RunningMode.VIDEO,num_poses=1,
        min_pose_detection_confidence=0.5,min_tracking_confidence=0.6)
    landmarker=PoseLandmarker.create_from_options(options)

    smoother=Smoother(alpha=SMOOTH_ALPHA)
    prev_pts={}
    histories={r:deque(maxlen=int(fps*8)) for r in REGION_LABELS}
    view_votes={'front':0,'side':0}
    stable_view='front'
    stable_findings, stable_recs = [], []
    avg_intensity = {r: 0.0 for r in REGION_LABELS}

    fidx=0; ts=0; dt=1000.0/fps

    while True:
        ret,frame=cap.read()
        if not ret: break

        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        mp_img=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
        result=landmarker.detect_for_video(mp_img,int(ts)); ts+=dt

        lm=result.pose_landmarks[0] if result.pose_landmarks and len(result.pose_landmarks)>0 else None
        pts=smoother.update(lm,w,h)

        mv=compute_movement(prev_pts,pts)
        for r in REGION_LABELS:
            histories[r].append(mv[r])
            avg_intensity[r] = 0.9*avg_intensity[r] + 0.1*mv[r]
        prev_pts=pts

        v=detect_view(pts)
        if fidx<30: view_votes[v]+=1
        elif fidx==30:
            stable_view='front' if view_votes['front']>=view_votes['side'] else 'side'
            print(f"  Detected view: {stable_view}")
        if fidx>=30: v=stable_view

        angles = compute_joint_angles(pts)

        if fidx % int(fps*2)==0 and fidx>=30:
            stable_findings, stable_recs = generate_assessment(v, angles, avg_intensity)

        draw_skeleton(frame, pts)
        draw_joint_angles(frame, pts, angles)
        if v=='front': draw_front_guides(frame, pts, angles)
        else: draw_side_guides(frame, pts)
        pb = draw_metrics_panel(frame, v, angles, w)
        if stable_findings:
            draw_assessment(frame, stable_findings, stable_recs, w, pb)
        draw_waveform(frame, histories, w, h)

        out.write(frame)
        fidx+=1
        if fidx%50==0: print(f"  {fidx}/{total} ({100*fidx//total}%)")

    cap.release(); out.release(); landmarker.close()
    print(f"Done: {output_path} ({fidx} frames)")


if __name__=="__main__":
    if len(sys.argv)>=3: process(sys.argv[1],sys.argv[2])
    else: process("/tmp/flekks-viz/input.mp4","/tmp/flekks-viz/pt_output.mp4")
