#!/usr/bin/env python3
"""
Flekks Review Agent — Deterministic Quality Checks for Movement Analysis

Reads the JSON output from backend_processor.py and produces a quality report.
Checks: tracking quality, angle sanity, ROM consistency, view detection,
clinical norm alignment, and waveform sanity.

Usage:
    python3 review_agent.py <analysis.json> [--output <report.json>]
    # If no --output, prints report to stdout

The report is designed to be read by an AI agent (or human) to decide
whether to re-process with different parameters.
"""

import json
import sys
import os
import math

# ── Anatomical limits (absolute physical maxima) ──
# Knee/Hip/Elbow use included-angle convention: 180 = straight, lower = more flexed
# Shoulder uses included-angle from gonio: 180 = arm overhead, 0 = arm at side
ANATOMICAL_LIMITS = {
    "leftKnee":      (30, 185),    # included angle: 180=straight, ~30=deep squat
    "rightKnee":     (30, 185),
    "leftHip":       (30, 185),    # included angle: 180=standing, ~60=deep hip flex
    "rightHip":      (30, 185),
    "leftElbow":     (30, 185),    # included angle: 180=straight arm
    "rightElbow":    (30, 185),
    "leftShoulder":  (0, 185),     # included angle: 180=arm overhead
    "rightShoulder": (0, 185),
    "spineTilt":     (0, 45),
    "hipShift":      (-50, 50),
    "leftKneeValgus":  (-30, 30),
    "rightKneeValgus": (-30, 30),
}

# ── AAOS clinical norms — expected ROM range (min-max span in degrees) ──
CLINICAL_NORMS = {
    "leftKnee":      {"name": "L Knee ROM",      "expected_range": (10, 140)},
    "rightKnee":     {"name": "R Knee ROM",      "expected_range": (10, 140)},
    "leftHip":       {"name": "L Hip ROM",       "expected_range": (10, 130)},
    "rightHip":      {"name": "R Hip ROM",       "expected_range": (10, 130)},
    "leftElbow":     {"name": "L Elbow ROM",     "expected_range": (5, 150)},
    "rightElbow":    {"name": "R Elbow ROM",     "expected_range": (5, 150)},
    "leftShoulder":  {"name": "L Shoulder ROM",  "expected_range": (10, 180)},
    "rightShoulder": {"name": "R Shoulder ROM",  "expected_range": (10, 180)},
}

# Max allowed angle jump between consecutive frames (degrees)
MAX_ANGLE_JUMP = 40.0

# Minimum tracking coverage (fraction of frames with required landmarks)
MIN_TRACKING_COVERAGE = 0.80


def decode_frame_track(data):
    """Reconstruct per-frame angle data from keyframe+delta compressed format."""
    keyframes = data.get("keyframes", [])
    deltas = data.get("deltas") or []

    # Build index: frameIndex -> angles
    frames = {}
    last_angles = {}

    for kf in keyframes:
        angles = kf["angles"]
        frames[kf["frameIndex"]] = angles.copy()
        last_angles = angles.copy()

    # Apply deltas on top of last keyframe
    kf_sorted = sorted(keyframes, key=lambda k: k["frameIndex"])

    # Build a mapping of kf intervals
    for delta in deltas:
        fi = delta["frameIndex"]
        changes = delta["changes"]
        # Find the preceding keyframe
        base = {}
        for kf in kf_sorted:
            if kf["frameIndex"] <= fi:
                base = kf["angles"].copy()
            else:
                break
        # Apply accumulated deltas from that keyframe to this delta
        # Actually, for our purposes the changes dict IS the changed values
        # The unchanged values remain at keyframe values
        merged = base.copy()
        merged.update(changes)
        frames[fi] = merged

    # Fill in gaps: for frames between keyframes, interpolate or carry forward
    total_frames = data.get("totalFrames", 0)
    all_frames = []
    current = {}
    for i in range(total_frames):
        if i in frames:
            current = frames[i].copy()
        all_frames.append(current.copy())

    return all_frames


def check_tracking_quality(all_frames, total_frames):
    """Check what fraction of frames have all required landmarks (non-default angles)."""
    required_keys = ["leftKnee", "rightKnee", "leftHip", "rightHip"]
    good_frames = 0
    missing_ranges = []
    in_gap = False
    gap_start = 0

    for i, angles in enumerate(all_frames):
        has_all = all(
            k in angles and angles[k] != 180.0  # 180 is default/missing
            for k in required_keys
        )
        if has_all:
            good_frames += 1
            if in_gap:
                missing_ranges.append((gap_start, i - 1))
                in_gap = False
        else:
            if not in_gap:
                gap_start = i
                in_gap = True

    if in_gap:
        missing_ranges.append((gap_start, len(all_frames) - 1))

    coverage = good_frames / max(total_frames, 1)
    return coverage, missing_ranges


def check_angle_sanity(all_frames):
    """Check for anatomically impossible angle values."""
    issues = []
    for i, angles in enumerate(all_frames):
        for key, (lo, hi) in ANATOMICAL_LIMITS.items():
            if key in angles:
                val = angles[key]
                if val < lo - 5 or val > hi + 5:  # 5 degree tolerance
                    issues.append({
                        "type": "impossible_angle",
                        "joint": key,
                        "frame": i,
                        "value": round(val, 1),
                        "limit": f"{lo}-{hi}",
                    })
    return issues


def check_angle_jumps(all_frames):
    """Check for sudden angle jumps that indicate tracking glitches."""
    issues = []
    angle_keys = list(ANATOMICAL_LIMITS.keys())

    for i in range(1, len(all_frames)):
        prev = all_frames[i - 1]
        curr = all_frames[i]
        for key in angle_keys:
            if key in prev and key in curr:
                delta = abs(curr[key] - prev[key])
                if delta > MAX_ANGLE_JUMP:
                    issues.append({
                        "type": "angle_jump",
                        "joint": key,
                        "frame": i,
                        "delta": round(delta, 1),
                        "threshold": MAX_ANGLE_JUMP,
                    })
    return issues


def check_view_detection(analysis):
    """Check view detection confidence."""
    view = analysis.get("detectedView", "unknown")
    confidence = analysis.get("viewConfidence", 0)
    return {
        "view": view,
        "confidence": round(confidence, 2),
        "reliable": confidence > 0.65,
    }


def check_rom_vs_norms(analysis):
    """Compare measured ROM range against expected ranges."""
    rom = analysis.get("rom", {})
    results = {}

    key_map = {
        "l_knee": "leftKnee", "r_knee": "rightKnee",
        "l_hip": "leftHip", "r_hip": "rightHip",
        "l_elbow": "leftElbow", "r_elbow": "rightElbow",
        "l_shoulder": "leftShoulder", "r_shoulder": "rightShoulder",
    }

    for internal_key, ios_key in key_map.items():
        if internal_key not in rom:
            continue
        spec = CLINICAL_NORMS.get(ios_key)
        if not spec:
            continue

        measured = rom[internal_key]
        measured_range = measured["range"]
        lo, hi = spec["expected_range"]

        if measured_range < lo:
            status = "limited"
        elif measured_range > hi:
            status = "excessive"
        else:
            status = "normal"

        results[internal_key] = {
            "measured": round(measured_range, 1),
            "expected": f"{lo}-{hi}",
            "status": status,
        }

    return results


def check_gait_metrics(analysis):
    """Check gait metrics against norms."""
    gait = analysis.get("gait", {})
    results = {}

    arm_ratio = gait.get("armSwingRatio", 1.0)
    if abs(arm_ratio - 1.0) > 0.8:
        results["armSwingRatio"] = {"measured": arm_ratio, "expected": "0.6-1.4", "status": "abnormal"}
    elif abs(arm_ratio - 1.0) > 0.4:
        results["armSwingRatio"] = {"measured": arm_ratio, "expected": "0.6-1.4", "status": "borderline"}
    else:
        results["armSwingRatio"] = {"measured": arm_ratio, "expected": "0.6-1.4", "status": "normal"}

    sw = gait.get("stepWidthCm", 0)
    if sw > 0:
        if sw > 20:
            results["stepWidthCm"] = {"measured": sw, "expected": "8-15cm", "status": "abnormal"}
        elif sw > 15:
            results["stepWidthCm"] = {"measured": sw, "expected": "8-15cm", "status": "borderline"}
        else:
            results["stepWidthCm"] = {"measured": sw, "expected": "8-15cm", "status": "normal"}

    return results


def check_waveform_sanity(data):
    """Check body channels for dead waveforms (tracking lost)."""
    channels = data.get("bodyChannels")
    if not channels or len(channels) == 0:
        return {"available": False}

    # Sum intensity per region across all frames
    region_sums = {"torso": 0, "leftArm": 0, "rightArm": 0, "leftLeg": 0, "rightLeg": 0}
    for frame in channels:
        for region in region_sums:
            region_sums[region] += frame.get(region, {}).get("intensity", 0)

    dead_regions = [r for r, s in region_sums.items() if s < 0.1]

    return {
        "available": True,
        "regionSums": {k: round(v, 2) for k, v in region_sums.items()},
        "deadRegions": dead_regions,
    }


def compute_overall_quality(tracking_coverage, angle_issues, jump_issues, view_info, waveform_info):
    """Compute overall quality score: good, fair, or poor."""
    problems = 0

    if tracking_coverage < 0.70:
        problems += 3
    elif tracking_coverage < MIN_TRACKING_COVERAGE:
        problems += 1

    problems += min(len(angle_issues), 5)   # cap contribution
    problems += min(len(jump_issues), 5)

    if not view_info.get("reliable", True):
        problems += 2

    dead = waveform_info.get("deadRegions", [])
    problems += len(dead) * 2

    if problems == 0:
        return "good"
    elif problems <= 4:
        return "fair"
    else:
        return "poor"


def review(data):
    """Run all quality checks on the output of backend_processor.py."""
    frame_track = data.get("frameTrack", {})
    analysis = data.get("analysis", {})
    total_frames = frame_track.get("totalFrames", 0)

    # Decode compressed frames
    all_frames = decode_frame_track(frame_track)

    # Run checks
    tracking_coverage, missing_ranges = check_tracking_quality(all_frames, total_frames)
    angle_issues = check_angle_sanity(all_frames)
    jump_issues = check_angle_jumps(all_frames)
    view_info = check_view_detection(analysis)
    rom_vs_norms = check_rom_vs_norms(analysis)
    gait_metrics = check_gait_metrics(analysis)
    waveform_info = check_waveform_sanity(data)

    overall = compute_overall_quality(tracking_coverage, angle_issues, jump_issues, view_info, waveform_info)

    # Collect all issues
    issues = []
    for gap_start, gap_end in missing_ranges:
        length = gap_end - gap_start + 1
        if length >= 3:
            issues.append({
                "type": "tracking_gap",
                "frames": list(range(gap_start, gap_end + 1))[:10],  # truncate for readability
                "description": f"Tracking lost for {length} frames ({gap_start}-{gap_end})",
            })

    issues.extend(angle_issues[:20])   # cap at 20
    issues.extend(jump_issues[:20])

    if not view_info.get("reliable", True):
        issues.append({
            "type": "view_ambiguous",
            "description": f"View detection only {view_info['confidence']:.0%} confident ({view_info['view']})",
        })

    for region in waveform_info.get("deadRegions", []):
        issues.append({
            "type": "dead_waveform",
            "region": region,
            "description": f"{region} has near-zero movement signal — tracking may be lost",
        })

    report = {
        "overallQuality": overall,
        "trackingCoverage": round(tracking_coverage, 3),
        "totalFrames": total_frames,
        "viewDetection": view_info,
        "issues": issues,
        "metricsVsNorms": {**rom_vs_norms, **gait_metrics},
        "waveformSanity": waveform_info,
    }

    return report


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 review_agent.py <analysis.json> [--output <report.json>]", file=sys.stderr)
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = None
    for i, arg in enumerate(sys.argv):
        if arg == "--output" and i + 1 < len(sys.argv):
            output_path = sys.argv[i + 1]

    if not os.path.exists(input_path):
        print(f"ERROR: File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    with open(input_path) as f:
        data = json.load(f)

    report = review(data)

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report written to {output_path}", file=sys.stderr)
        print(f"Quality: {report['overallQuality']} | Tracking: {report['trackingCoverage']:.1%} | Issues: {len(report['issues'])}", file=sys.stderr)
    else:
        json.dump(report, sys.stdout, indent=2)
        print(file=sys.stderr)
        print(f"Quality: {report['overallQuality']} | Tracking: {report['trackingCoverage']:.1%} | Issues: {len(report['issues'])}", file=sys.stderr)


if __name__ == "__main__":
    main()
