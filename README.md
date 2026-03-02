# Flekks Movement Analysis Pipeline

Video-based movement analysis using MediaPipe pose estimation with clinical-grade goniometric joint angle measurements, gait metrics, and movement intensity waveforms.

**Repo**: `Dylan-Flekks/flekks-movement-analysis`

---

## Table of Contents

- [Quick Start](#quick-start)
- [Scripts](#scripts)
- [API Index](#api-index)
  - [gait_analyzer.py](#gait_analyzerpy-api)
  - [pt_analyzer.py](#pt_analyzerpy-api)
  - [process.py](#processpy-api)
  - [posture_analyzer.py](#posture_analyzerpy-api)
  - [squat_analyzer.py](#squat_analyzerpy-api)
- [MediaPipe Landmark Map](#mediapipe-landmark-map)
- [Goniometric Conventions](#goniometric-conventions)
- [Clinical Norms](#clinical-norms)
- [Metric Catalog](#metric-catalog)
- [Visual Overlays](#visual-overlays)
- [Architecture](#architecture)
- [Video Catalog](#video-catalog)
- [Version History](#version-history)
- [Infrastructure](#infrastructure)
- [Design Decisions](#design-decisions)
- [References](#references)

---

## Quick Start

```bash
pip install opencv-python mediapipe numpy

# Download model
wget -O pose_landmarker_heavy.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task

# Run gait analysis (latest)
python3 gait_analyzer.py input.mp4 output.mp4

# Run PT clinical analysis
python3 pt_analyzer.py input.mp4 output.mp4

# Run movement waveform only
python3 process.py input.mp4 output.mp4
```

**Dependencies**: Python 3.10+, OpenCV 4.11.0, MediaPipe 0.10.21, NumPy

---

## Scripts

| Script | Purpose | View | Coords | Latest |
|---|---|---|---|---|
| `gait_analyzer.py` | Gait analysis, goniometric angles, plane-specific panels | Auto (front/side) | 3D world | Yes |
| `pt_analyzer.py` | Full PT/clinical analysis, NL assessment, diagnostic recs | Auto (front/side) | 2D | v6 |
| `process.py` | Movement intensity waveforms only | Any | 2D | v7 |
| `posture_analyzer.py` | Side-profile postural analysis | Side | 2D | v2 |
| `squat_analyzer.py` | Frontal-plane squat analysis | Front | 2D | v2 |

### `gait_analyzer.py` (LATEST)

Full gait analysis using **3D world landmarks** for all angle calculations. Auto-detects frontal vs sagittal view.

- Goniometric reference frames (trunk axis) for shoulder flex, hip ABD, shoulder ABD
- 3D angle computation via `angle_at_3d()` and `angle_between_vectors()`
- Plane-specific panels (frontal: ABD, valgus, circumduction, eversion; sagittal: flex/ext, ROM, CVA)
- AutoCAD-style arc indicators with FLEX/EXT labels
- ROM tracking (rolling 3s min/max window)
- Gait-specific trackers (circumduction, hip hiking, arm swing ratio, step width)
- Movement intensity waveforms per body region

### `pt_analyzer.py` (v6)

Combined PT analyzer with data-driven natural language assessment.

- AutoCAD arc indicators at each joint with flexion/extension labels
- Plain English "WHAT THIS MEANS" section with layman-friendly findings
- Diagnostic "NEXT:" recommendations (what angle to film next)
- Clinical norms for all measurements (AAOS standards)
- Kyphosis, lordosis, pelvic tilt (side view)
- FPPA knee valgus, hip shift, shoulder tilt (front view)

### `process.py` (v7)

Pure movement visualization.

- Per-limb overlapping waveforms with glow lines (no fill)
- Dynamic z-ordering (most active region drawn on top)
- Minimal skeleton overlay with Flekks brand colors

### `posture_analyzer.py` (v2)

Side-profile postural analysis.

- CVA (craniovertebral angle) with plumb line
- Thoracic kyphosis estimate (ear-shoulder-hip deviation from 180)
- Lumbar lordosis estimate (shoulder-hip-knee deviation from 180)
- Anterior pelvic tilt (hip-to-knee angle from vertical)
- Forward shoulder protraction
- Overall posture score (0-100)

### `squat_analyzer.py` (v2)

Front-view squat symmetry analysis.

- FPPA (frontal plane projection angle) for knee valgus/varus
- Ankle pronation (heel-to-ankle lateral shift)
- Hip shift (lateral displacement normalized to shoulder width)
- Shoulder tilt, depth asymmetry
- Squat depth tracking with graph

---

## API Index

### `gait_analyzer.py` API

#### Classes

| Class | Description |
|---|---|
| `Smoother(alpha, n)` | EMA smoother for 2D `pose_landmarks`. Returns `{idx: (px, py)}` |
| `WorldSmoother(alpha, n)` | EMA smoother for 3D `pose_world_landmarks`. Returns `{idx: (x, y, z)}` in meters |
| `ROMTracker(window)` | Rolling min/max/current angle tracker over `window` frames |
| `GaitTracker(window)` | Hemiparesis-specific gait indices from 2D pixel positions |

#### ROMTracker Methods

| Method | Returns |
|---|---|
| `update(angles)` | Stores angles for tracked joints |
| `get_rom(key)` | `(min, max, current)` or `None` if < 5 samples |

#### GaitTracker Methods

| Method | Returns | Description |
|---|---|---|
| `update(pts, world_pts)` | — | Feed 2D + 3D points each frame |
| `circumduction_index(side)` | `float` (px) | Lateral ankle excursion range. 'l' or 'r' |
| `hip_hiking()` | `(l_range, r_range, obliq)` (px) | Vertical hip excursion + asymmetry |
| `arm_swing_ratio()` | `float` | L/R arm swing ratio. 1.0 = symmetric |
| `avg_step_width()` | `float` (meters) | Mean lateral ankle distance from 3D coords |

#### Geometry Functions

| Function | Signature | Description |
|---|---|---|
| `mid(a, b)` | `(tuple, tuple) -> (int, int)` | 2D midpoint |
| `angle_at(p1, v, p2)` | `(2D, 2D, 2D) -> float` | 2D included angle at vertex v |
| `angle_at_3d(p1, v, p2)` | `(3D, 3D, 3D) -> float` | 3D included angle at vertex v (0-180) |
| `compute_ankle_df_pf_3d(knee, ankle, foot)` | `(3D, 3D, 3D) -> float` | DF/PF: positive=DF, negative=PF, 0=neutral |
| `angle_from_vertical_3d(top, bottom)` | `(3D, 3D) -> float` | Segment deviation from vertical (y-axis) |
| `normalize_vec(v)` | `ndarray -> ndarray` | Safe unit vector (returns zeros if too small) |
| `trunk_axis(world_pts)` | `dict -> ndarray or None` | Unit vector from mid-hip to mid-shoulder |
| `angle_between_vectors(v1, v2)` | `(ndarray, ndarray) -> float` | Angle between two 3D vectors in degrees |

#### Goniometric Functions

| Function | Signature | 0 Position | Description |
|---|---|---|---|
| `gonio_shoulder_flex(wp, side)` | `(dict, 'l'/'r') -> float` | Arm at side | Angle between -trunk_axis and upper arm |
| `gonio_hip_abd(wp, side)` | `(dict, 'l'/'r') -> float` | Legs together | Angle between -trunk_axis and femur |
| `gonio_shoulder_abd(wp, side)` | `(dict, 'l'/'r') -> float` | Arm at side | Same as shoulder flex (3D) |

#### Core Functions

| Function | Description |
|---|---|
| `detect_view(pts, frame_w)` | Binary front/side classification from shoulder+hip separation |
| `compute_movement(prev, curr)` | Per-region pixel displacement between frames |
| `compute_joint_angles(world_pts, pts_2d)` | All angle computations (3D world + 2D overlay metrics) |
| `process(input_path, output_path)` | Main entry point: video in, annotated video out |

#### Drawing Functions

| Function | Description |
|---|---|
| `draw_skeleton(f, pts)` | Skeleton with glow effect (outer dim + inner bright + white dots) |
| `draw_arc(f, vertex, p1, p2, angle, color, radius, thickness)` | AutoCAD-style arc between limb segments |
| `draw_joint_angles(f, pts, angles, view)` | Plane-appropriate arcs + FLEX/EXT labels |
| `draw_front_guides(f, pts, angles)` | FPPA arrows + hip shift plumb line |
| `draw_frontal_panel(f, angles, rom, gait, w)` | Right-side frontal metrics panel |
| `draw_sagittal_panel(f, angles, rom, gait, w)` | Right-side sagittal metrics panel |
| `draw_waveform(f, histories, w, h, wave_h)` | Bottom movement intensity waveform |
| `get_flexion_label(key, angle, angles)` | Human-readable label: "FLEX 30", "EXT", "DF 15", "NEUT" |
| `get_ankle_label(dfpf_val)` | "DF X", "PF X", or "NEUT" |

### `pt_analyzer.py` API

| Function | Description |
|---|---|
| `compute_joint_angles(pts)` | All joint angles from 2D points (no world landmarks) |
| `generate_assessment(view, angles, intensity)` | Data-driven NL findings + NEXT: recommendations |
| `draw_joint_angles(f, pts, angles)` | Arcs + labels (all joints, no view filtering) |
| `draw_metrics_panel(f, view, angles, w)` | JOINT ANALYSIS (front) or POSTURE ANALYSIS (side) panel |
| `draw_assessment(f, findings, recs, w, panel_top)` | "WHAT THIS MEANS" plain English section |
| `get_flexion_label(key, angle_deg)` | Label without angles dict (simpler version) |

### `process.py` API

| Function | Description |
|---|---|
| `LandmarkSmoother.smooth(landmarks, w, h)` | EMA smoothing, returns `{idx: (px, py)}` |
| `compute_region_movement(prev, curr)` | Per-region pixel displacement |
| `draw_skeleton(frame, points)` | Minimal skeleton with brand colors |
| `draw_overlapping_waveform(frame, histories, w, h, wave_height)` | Glow-line waveforms with dynamic z-ordering |

### `posture_analyzer.py` API

| Function | Description |
|---|---|
| `pick_side(points)` | Select more visible side for analysis |
| `compute_posture(points)` | CVA, kyphosis, lordosis, pelvic tilt, shoulder protraction, score |
| `draw_posture_overlay(frame, metrics, w, h)` | Plumb line + posture chain + angle labels |
| `draw_metrics_panel(frame, metrics, w, h)` | POSTURE ANALYSIS panel with findings |

### `squat_analyzer.py` API

| Function | Description |
|---|---|
| `compute_fppa(hip, knee, ankle)` | FPPA angle + cross product for valgus/varus direction |
| `compute_squat_metrics(points)` | All squat metrics (FPPA, pronation, hip shift, depth) |
| `draw_valgus_overlay(frame, points, metrics)` | Ideal alignment lines + valgus arrows |
| `draw_pronation_indicators(frame, points, metrics)` | Ankle pronation indicators |
| `draw_depth_graph(frame, depth_history, w, h)` | Bottom-left squat depth graph |

---

## MediaPipe Landmark Map

```
Landmark indices used across all scripts:

  0  NOSE
  7  LEFT_EAR          8  RIGHT_EAR
 11  LEFT_SHOULDER     12  RIGHT_SHOULDER
 13  LEFT_ELBOW        14  RIGHT_ELBOW
 15  LEFT_WRIST        16  RIGHT_WRIST
 23  LEFT_HIP          24  RIGHT_HIP
 25  LEFT_KNEE         26  RIGHT_KNEE
 27  LEFT_ANKLE        28  RIGHT_ANKLE
 29  LEFT_HEEL         30  RIGHT_HEEL
 31  LEFT_FOOT_INDEX   32  RIGHT_FOOT_INDEX
```

**Skeleton connections** (drawn as bone segments):
```
Torso:     11-12, 11-23, 12-24, 23-24
Arms:      11-13, 13-15, 12-14, 14-16
Legs:      23-25, 25-27, 24-26, 26-28
Feet:      27-29, 28-30, 27-31, 28-32
```

**Region groupings** (for movement waveforms):
| Region | Landmarks |
|---|---|
| Torso | 11, 12, 23, 24 |
| Left Arm | 11, 13, 15 |
| Right Arm | 12, 14, 16 |
| Left Leg | 23, 25, 27 |
| Right Leg | 24, 26, 28 |

**Coordinate systems**:
- `pose_landmarks` (2D): normalized [0,1] x,y relative to image. Multiplied by (w,h) for pixel coords.
- `pose_world_landmarks` (3D): meters, pelvis-centered. x = lateral, y = vertical (down), z = depth (toward camera).

---

## Goniometric Conventions

### Included-Angle Joints (knee, hip, elbow)

Raw `angle_at_3d()` returns the geometric included angle (0-180). Full extension = 180. Goniometric flexion = `180 - included_angle`.

| Joint | Computation | Example |
|---|---|---|
| Knee FLEX | `180 - angle_at_3d(HIP, KNEE, ANKLE)` | 180 = full ext, 60 = deep bend |
| Hip FLEX | `180 - angle_at_3d(SHOULDER, HIP, KNEE)` | 180 = standing, 30 = swing phase |
| Elbow FLEX | `180 - angle_at_3d(SHOULDER, ELBOW, WRIST)` | 180 = straight arm, 90 = right angle |

### Trunk-Referenced Joints (shoulder flex, shoulder ABD, hip ABD)

Uses `angle_between_vectors()` against the trunk axis. Already in goniometric convention (0 = anatomical position).

| Joint | Computation | 0 | 90 |
|---|---|---|---|
| Shoulder FLEX | `angle(-trunk_up, shoulder->elbow)` | Arm at side | Arm horizontal |
| Shoulder ABD | `angle(-trunk_up, shoulder->elbow)` | Arm at side | Arm horizontal |
| Hip ABD | `angle(-trunk_up, hip->knee)` | Legs together | N/A (max ~45) |

**Trunk axis**: `normalize(mid_shoulder - mid_hip)` — unit vector pointing up along the trunk.

### Ankle DF/PF

```
angle = angle_at_3d(KNEE, ANKLE, FOOT_INDEX)
DF/PF = 90 - angle
```
- Positive = dorsiflexion (foot toward shin)
- Negative = plantarflexion (foot pointing away)
- Zero = neutral (foot 90 to shin)

### Ankle Eversion/Inversion

Heel-to-foot_index vector projected into frontal (x-y) plane. Eversion = foot sole faces outward (positive).

---

## Clinical Norms

### AAOS ROM Standards

| Measurement | Normal Range | Source |
|---|---|---|
| Knee Flexion | 0-135 | AAOS |
| Hip Flexion | 0-120 | AAOS |
| Elbow Flexion | 0-145 | AAOS |
| Shoulder Flexion | 0-180 | AAOS |
| Shoulder ABD | 0-180 | AAOS |
| Hip ABD | 0-45 | AAOS |
| Ankle DF | 0-20 | AAOS |
| Ankle PF | 0-50 | AAOS |

### Postural Norms

| Measurement | Normal | Warning | Abnormal |
|---|---|---|---|
| CVA | >= 50 | 45-50 | < 45 (FHP) |
| Thoracic Kyphosis | 20-40 | 15-20 or 40-45 | < 15 or > 45 |
| Lumbar Lordosis | 30-50 | 25-30 or 50-55 | < 25 or > 55 |
| Pelvic Tilt | 7-15 | 5-7 or 15-18 | < 5 or > 18 |
| Trunk Lean | 0-8 | 8-12 | > 12 |

### Frontal Plane Norms

| Measurement | Normal | Mild | Significant |
|---|---|---|---|
| FPPA (knee valgus) | < 8 | 8-12 | > 12 |
| Hip Shift | < 3% | 3-8% | > 8% |
| Shoulder Tilt | < 2 | 2-4 | > 4 |
| Ankle Pronation | < 2% | 2-5% | > 5% |

### Gait Norms

| Measurement | Normal | Warning | Source |
|---|---|---|---|
| Step Width | 8-12 cm | > 15 cm | 3D world coords |
| Arm Swing Ratio | 0.6-1.4x | < 0.6 or > 1.4 | L/R pixel displacement |
| Circumduction | < 40 px | > 60 px | Lateral ankle excursion |

---

## Metric Catalog

### Sagittal Panel (`gait_analyzer.py`, side view)

| Row | Key(s) | Unit | Description |
|---|---|---|---|
| KNEE L/R | `l_knee`, `r_knee` | deg | Flexion + rolling ROM |
| HIP L/R | `l_hip`, `r_hip` | deg | Flexion + rolling ROM |
| ANKLE | `l_ankle_dfpf`, `r_ankle_dfpf` | deg | DF/PF from 90 neutral |
| SHLDR L/R | `l_shoulder`, `r_shoulder` | deg | Goniometric flexion (0=neutral) |
| TRUNK LEAN | `trunk` | deg | Trunk deviation from vertical |
| CVA | `cva` | deg | Craniovertebral angle |
| ARM SWING | via `GaitTracker` | ratio | L/R arm swing symmetry |

### Frontal Panel (`gait_analyzer.py`, front view)

| Row | Key(s) | Unit | Description |
|---|---|---|---|
| HIP ABD | `l_hip_abd`, `r_hip_abd` | deg | Hip abduction (trunk-referenced) |
| VALGUS | `l_fppa`, `r_fppa` | deg | Knee valgus/varus (FPPA) |
| CIRCUMDUCT | via `GaitTracker` | px | Lateral ankle excursion range |
| ANKLE E/I | `l_ev_inv`, `r_ev_inv` | deg | Eversion/inversion |
| PELVIS | `pelv_obliq` | px | Pelvic obliquity (L/R height diff) |
| HIP HIKE | via `GaitTracker` | px | Vertical hip excursion |
| SH ABD | `l_sh_abd`, `r_sh_abd` | deg | Shoulder abduction (trunk-referenced) |
| ARM SWING | via `GaitTracker` | ratio | L/R symmetry |
| STEP WIDTH | via `GaitTracker` | cm | Lateral ankle distance (3D) |
| SH TILT | `sh_tilt` | deg | Shoulder height asymmetry |

### PT Analyzer Metrics (`pt_analyzer.py`)

| Metric | Key | View | Description |
|---|---|---|---|
| Knee flex | `l_knee`, `r_knee` | Both | 2D included angle |
| Hip flex | `l_hip`, `r_hip` | Both | 2D included angle |
| Elbow flex | `l_elbow`, `r_elbow` | Both | 2D included angle |
| Shoulder flex | `l_shoulder`, `r_shoulder` | Both | 2D included angle |
| Ankle | `l_ankle`, `r_ankle` | Both | 2D KNEE-ANKLE-FOOT angle |
| FPPA | `l_fppa`, `r_fppa` | Front | Knee valgus/varus |
| Hip shift | `hip_shift` | Front | % of shoulder width |
| Shoulder tilt | `sh_tilt` | Front | Degrees |
| Pelvic drop | `pelv_drop` | Front | % of torso height |
| CVA | `cva` | Side | Craniovertebral angle |
| Kyphosis | `kyphosis` | Side | Upper back rounding |
| Lordosis | `lordosis` | Side | Low back curve |
| Trunk lean | `trunk` | Side | Forward lean |
| All asymmetries | `*_asym` | Both | |L - R| for each joint |

---

## Visual Overlays

### Skeleton
- Outer glow (dim accent, 6px) + inner line (bright accent, 2px)
- Joint dots: outer dim (6px) + mid accent (4px) + white core (2px)

### Arc Indicators
- AutoCAD-style curved arc between parent and child bone segments
- Tick marks at arc endpoints (dimension-line feel)
- Leader line from arc to label
- Color by L/R asymmetry: cyan (< 6), orange/yellow (6-12), red (> 12)
- Dimmed for near-extension (> 170)

### Hip ABD/ADD Arc Indicators (Frontal View)
- Vertical reference line drawn downward from each hip joint
- Arc drawn between vertical reference and femur direction
- Signed values: positive = ABD (away from midline), negative = ADD (toward midline)
- Labels positioned well outside body (55% of femur length laterally) to avoid overlap
- Leader line from arc to pill label
- Labels: "ABD X°" / "NEUT" / "ADD X°" (threshold ±3°)
- Color by L/R asymmetry (same scheme as flex/ext arcs)

### Labels
- Pill-shaped dark background (80% opacity) with colored border
- FLEX/EXT/DF/PF/NEUT prefix + degrees
- Shoulder labels: "NEUT" when < 5, "FLEX X" otherwise (already goniometric)
- Hip ABD/ADD labels: positioned laterally outside body to prevent overlap with skeleton

### Waveform
- Per-region (torso, L arm, R arm, L leg, R leg) as overlapping glow lines
- Gaussian-smoothed (k=11)
- Dynamic z-ordering: most active region drawn on top
- Vertical playhead at current frame
- Color legend at top of waveform area

### Frontal Guides
- Hip-to-ankle plumb line (gray)
- Knee deviation arrows (green/yellow/red based on FPPA)
- Shoulder-to-hip midpoint plumb + hip shift arrow

### Color Scheme (BGR)

| Name | BGR | Use |
|---|---|---|
| ACCENT | (166, 180, 64) | Skeleton, headers |
| ACCENT_DIM | (115, 125, 45) | Skeleton glow |
| WHITE | (240, 245, 240) | Joint cores, text |
| RED | (70, 70, 230) | Abnormal values |
| GREEN | (100, 200, 100) | Normal values |
| YELLOW | (60, 200, 230) | Warning values |
| CYAN | (200, 180, 50) | Normal arc/label |
| ORANGE | (50, 140, 230) | Moderate asymmetry |
| DARK_BG | (15, 15, 15) | Panel/label backgrounds |

---

## Architecture

### Processing Pipeline

```
Video Frame
    |
    v
MediaPipe PoseLandmarker (heavy model, VIDEO mode)
    |
    +-- pose_landmarks (2D normalized)
    |       |
    |       v
    |   Smoother (EMA alpha=0.45)
    |       |
    |       v
    |   {idx: (px, py)} -- used for drawing + 2D metrics
    |
    +-- pose_world_landmarks (3D meters)
            |
            v
        WorldSmoother (EMA alpha=0.45)
            |
            v
        {idx: (x, y, z)} -- used for all angle math
            |
            v
        compute_joint_angles()
            |
            +-- Goniometric functions (trunk-referenced)
            +-- Included-angle functions (angle_at_3d)
            +-- 2D overlay metrics (FPPA, hip shift)
            |
            v
        ROMTracker + GaitTracker
            |
            v
        View-specific drawing
            |
            v
        Output Frame
```

### View Detection
- Binary front/side based on average shoulder+hip lateral pixel separation
- Threshold: 15% of frame width
- Voted over first 30 frames, then locked for rest of video

### Smoothing
- Exponential moving average: `state = alpha * state + (1-alpha) * raw`
- Alpha = 0.45 (higher = smoother, more lag)
- Visibility decay: when landmarks lost, visibility *= 0.85 per frame
- Minimum visibility threshold: 0.3 (2D) or 0.3 (3D)

---

## Video Catalog

All source videos from Mixkit (720p free stock).

### Front View
- `squat_home.mp4`, `squat_kettlebell.mp4`, `squat_woman.mp4`, `squat_man.mp4`
- `squat_toned.mp4`, `squat_woman_db.mp4`, `squat_diagonal.mp4`
- `lunge_front.mp4`

### Side/Profile View
- `walking_profile.mp4`, `walking_steps.mp4`, `walker_stadium.mp4`
- `squat_profile.mp4`, `deadlift_side.mp4`, `yoga_dog.mp4`
- `jump_lunge.mp4`

### Diagonal/Multi-Angle
- `deadlift_woman.mp4` (rear diagonal, 50fps)
- `overhead_press.mp4` (front diagonal)
- `lunge_diagonal.mp4` (walking lunges)
- `pushup_ground.mp4` (ground-level)

### Martial/Movement
- `martial.mp4`, `karate_kick.mp4`, `karate_moves.mp4`
- `woman_karate.mp4`, `dance_spin.mp4`
- `boxer_dark.mp4`, `boxer_bag.mp4`, `boxer_silhouette.mp4`
- `baseball_pitch.mp4`, `kickboxer_female.mp4`

---

## Version History

| Ver | Script | Changes |
|---|---|---|
| v1-v5 | `process.py` | Iterated waveform viz: stacked, layered, overlapping fills, glow lines |
| v6 | `process.py` | Overlapping waveforms with semi-transparent fills |
| v7 | `process.py` | Lines only (no fill), glow effect, dynamic z-ordering |
| v1 | `squat_analyzer.py` | Basic squat symmetry analysis |
| v2 | `squat_analyzer.py` | Clinical norms (FPPA thresholds, hip shift %) |
| v1 | `posture_analyzer.py` | Basic posture metrics |
| v2 | `posture_analyzer.py` | Goniometric standards (CVA, kyphosis, lordosis, pelvic tilt) |
| v3 | `pt_analyzer.py` | Combined analyzer, auto view detection |
| v4 | `pt_analyzer.py` | Top-right metrics panel + plain English summary |
| v5 | `pt_analyzer.py` | Joint angles on skeleton, data-driven NL, diagnostic recs |
| v6 | `pt_analyzer.py` | AutoCAD arcs, FLEX/EXT labels, polished styling |
| v1 | `gait_analyzer.py` | 3D world landmarks, ankle FOOT_INDEX, view-specific panels |
| v2 | `gait_analyzer.py` | Goniometric trunk-axis refs, eversion/inversion, CVA 3D, step width |

---

## Infrastructure

- **Server**: `100.102.38.30` (Tailscale: `adha`), RTX 4060 Ti
- **Mac**: `100.74.247.45` (Tailscale), MacBook Pro
- **Processing**: CPU-based MediaPipe (not GPU), ~1-3 min per video
- **Model**: `pose_landmarker_heavy.task` — highest accuracy MediaPipe pose model
- **Video source**: Mixkit free stock videos (720p)
- **Model path**: `/tmp/flekks-viz/pose_landmarker_heavy.task`

---

## Design Decisions

Decisions made based on user feedback during development:

| Issue | Decision |
|---|---|
| Fill obscured line colors in waveform | Removed fills, used glow lines only |
| Raw degrees confusing to users | Added FLEX/EXT/DF/PF labels |
| LLM inference wildly inaccurate on some videos | Made all NL assessment data-driven from measured angles |
| Wanted "what to ask patient next" | Added diagnostic NEXT: recommendations |
| Wanted joint angles visible ON the skeleton | AutoCAD arc indicators + labels at each joint |
| Wanted 3D joint angles "like AutoCAD" | Arc + tick mark dimension line style |
| Shoulder flex used hip-to-shoulder diagonal as reference | Fixed: trunk axis (mid-hip to mid-shoulder) |
| Hip ABD used absolute vertical | Fixed: pelvis midline (trunk axis pointing down) |
| Ankle DF/PF used HEEL landmark (wrong geometry) | Fixed: uses FOOT_INDEX (31/32) landmark |
| User prefers layman terms but muscle names OK | Plain English + muscle names in parentheses |
| Intensity waveform must always be present | Waveform drawn in all scripts |

---

## References

- [Shoulder Flexion Goniometry - Brookbush Institute](https://brookbushinstitute.com/videos/shoulder-flexion-goniometry)
- [Goniometry: Shoulder Abduction - Physiopedia](https://www.physio-pedia.com/Goniometry:_Shoulder_Abduction)
- [Goniometry: Hip Abduction - Physiopedia](https://www.physio-pedia.com/Goniometry:_Hip_Abduction)
- [Goniometer - StatPearls / NCBI](https://www.ncbi.nlm.nih.gov/books/NBK558985/)
- [MediaPipe Pose Landmarks](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md)
- [AAOS Clinical ROM Standards](https://www.aaos.org/)
- [FPPA Knee Valgus - Journal of Orthopaedic & Sports Physical Therapy](https://www.jospt.org/)
