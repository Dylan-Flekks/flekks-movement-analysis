# Flekks Movement Visualization — Session Context

## Overview
Built a suite of video-based movement analysis tools for the Flekks fitness app. These process exercise/movement videos with MediaPipe pose estimation and overlay clinical-grade joint angle measurements, movement intensity waveforms, and natural language PT assessments.

## Scripts (all at `/tmp/flekks-viz/` on server `100.102.38.30`)

### `pt_analyzer.py` — v6 (LATEST, main tool)
**Combined PT Analyzer** — the primary tool. Auto-detects front vs side camera view.

Features:
- **AutoCAD-style arc indicators** at each joint — curved arcs between limb segments showing angle origin, with tick marks like dimension lines
- **Flexion/Extension labels** — `FLEX 85°`, `EXT`, `DF 15°` (dorsiflexion), `PF` (plantarflexion) instead of raw degrees
- **Leader lines** from arc to label
- **Pill-shaped label backgrounds** with color-coded borders (cyan=normal, orange=moderate asymmetry, red=significant)
- **VALGUS/VARUS labels** spelled out at knees
- **Top-right metrics panel** — clinical measurements with accent bar styling
- **Plain English "WHAT THIS MEANS"** section below metrics — layman-friendly findings
- **Diagnostic "NEXT:" recommendations** — what angle to film next based on findings
- **Intensity waveform** at bottom — per-limb movement (overlapping glow lines, dynamic z-ordering)
- **Data-driven assessment** — all natural language driven by actual measured joint angles + intensity, not assumptions

Joints measured (both sides):
- Elbow, Shoulder, Hip, Knee, Ankle
- FPPA (Frontal Plane Projection Angle) for knee valgus/varus
- CVA (Craniovertebral Angle) for forward head posture
- Kyphosis (upper back rounding), Lordosis (low back curve)
- Trunk lean, Hip shift, Shoulder tilt, Pelvic drop
- L/R asymmetries for all bilateral joints

Clinical norms used:
- CVA: normal ≥50°, FHP <48°
- Thoracic kyphosis: 20-40°
- Lumbar lordosis: 30-50°
- Anterior pelvic tilt: 7-15°
- FPPA knee valgus: normal <8°, moderate 8-12°, significant >12°

Usage: `python3 pt_analyzer.py input.mp4 output.mp4`

### `process.py` — v7 (movement waveform only)
Pure movement intensity visualization. Per-limb overlapping waveforms with glow lines (no fill), dynamic z-ordering. Minimal skeleton overlay with brand colors.

Usage: `python3 process.py input.mp4 output.mp4`

### `squat_analyzer.py` — v2 (standalone, superseded by pt_analyzer)
Front-view squat analysis: hip shift, knee valgus/varus, shoulder tilt.

### `posture_analyzer.py` — v2 (standalone, superseded by pt_analyzer)
Side-profile posture analysis: CVA, kyphosis, lordosis, pelvic tilt.

## Dependencies
- Python 3, OpenCV (`cv2`), MediaPipe (`mediapipe`), NumPy
- Model file: `/tmp/flekks-viz/pose_landmarker_heavy.task` (MediaPipe heavy model)
- Install: `pip install opencv-python mediapipe numpy`

## Videos on Server (`/tmp/flekks-viz/`)

### Original Input Videos
**Front view squats:**
- `squat_home.mp4`, `squat_kettlebell.mp4`, `squat_woman.mp4`, `squat_man.mp4`
- `squat_toned.mp4`, `squat_woman_db.mp4`, `squat_diagonal.mp4`

**Side profile:**
- `walking_profile.mp4`, `walking_steps.mp4`, `walker_stadium.mp4`, `squat_profile.mp4`

**Martial/movement:**
- `martial.mp4`, `karate_kick.mp4`, `karate_moves.mp4`, `woman_karate.mp4`, `dance_spin.mp4`

**Boxing/sports:**
- `boxer_dark.mp4`, `boxer_bag.mp4`, `boxer_silhouette.mp4`, `baseball_pitch.mp4`, `kickboxer_female.mp4`

**New multi-angle (downloaded 2026-02-25):**
- `deadlift_side.mp4` — man, diagonal/side-rear angle
- `deadlift_woman.mp4` — woman, rear diagonal, 50fps/1070 frames
- `pushup_ground.mp4` — ground-level low angle
- `yoga_dog.mp4` — upward/downward dog, side view
- `overhead_press.mp4` — barbell overhead, front/diagonal
- `jump_lunge.mp4` — explosive jumping lunges, side
- `lunge_diagonal.mp4` — walking lunges, diagonal angle
- `lunge_front.mp4` — front view lunges

### Processed v6 Results (latest)
All `*_v6.mp4` files — 12 videos processed with AutoCAD arc indicators + flexion/extension labels.

### Processed v7 Results (waveform only)
All `*_v7.mp4` files — pure movement intensity waveform overlay.

## Version History

| Version | Script | Key Changes |
|---------|--------|-------------|
| v1-v5 | process.py | Iterated waveform viz: stacked → layered → overlapping fills → glow lines |
| v6 | process.py | Overlapping waveforms with semi-transparent fills |
| v7 | process.py | Lines only (no fill), glow effect, dynamic z-ordering |
| v1 | squat_analyzer.py | Basic squat symmetry analysis |
| v2 | squat_analyzer.py | Clinical norms (FPPA, hip shift) |
| v1 | posture_analyzer.py | Basic posture metrics |
| v2 | posture_analyzer.py | Goniometric standards (CVA, kyphosis, lordosis) |
| v3 | pt_analyzer.py | Combined analyzer, auto view detection |
| v4 | pt_analyzer.py | Top-right metrics panel + plain English summary |
| v5 | pt_analyzer.py | Joint angles on skeleton, data-driven NL, diagnostic recs |
| v6 | pt_analyzer.py | **AutoCAD arcs, FLEX/EXT labels, polished styling** |

## User Feedback / Design Decisions
- Fill obscured line colors → removed fills, used glow lines
- Raw degrees confusing → added FLEX/EXT/DF/PF labels
- "LLM inference wildly inaccurate on some" → made all NL data-driven from measured angles
- Wanted "what to ask patient next" → diagnostic NEXT: recommendations
- Wanted joint angles visible ON the skeleton → arc indicators + labels at each joint
- Wanted 3D joint angles "like AutoCAD" → arc + tick mark dimension line style
- User prefers layman terms but muscle names OK
- Intensity waveform must always be present
- Keep pulling diverse videos from different camera angles

## Infrastructure
- **Server**: `100.102.38.30` (Tailscale: `adha`), RTX 4060 Ti
- **Mac**: `100.74.247.45` (Tailscale), MacBook Pro
- **Processing**: CPU-based MediaPipe (not GPU), ~1-3 min per video depending on length
- **Model**: `pose_landmarker_heavy.task` — most accurate MediaPipe pose model
- **Video source**: Mixkit free stock videos (720p)

## Next Steps / Ideas
- Handle diagonal/3-quarter camera angles better (current auto-detect is binary front/side)
- Multi-person pose tracking
- Real-time processing from phone camera
- Integration with Flekks iOS app `MovementIntensityTracker` + `BeatPulseGenerator`
- Generate music that matches movement intensity (Suno integration)
- Comparison view: before/after or side-by-side of two takes
