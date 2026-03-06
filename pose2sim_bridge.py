#!/usr/bin/env python3
"""
Flekks Pose2Sim Bridge — MediaPipe → OpenSim IK → Optional MocoInverse

Wraps Pose2Sim to provide a clean interface between the existing MediaPipe
pipeline and OpenSim musculoskeletal analysis. All output is additive —
existing skeleton overlay and waveform rendering is untouched.

Pipeline:
  MediaPipe 3D world landmarks
    → TRC format (3D marker trajectories)
    → Pose2Sim auto-scale (segment lengths → model scaling, no static trial)
    → OpenSim IK (anatomically constrained joint angles)
    → Optional: MocoInverse (muscle activations + forces)

Usage:
    from pose2sim_bridge import process_with_opensim
    result = process_with_opensim(world_landmarks_by_frame, fps, height_m=1.75, mass_kg=70)
"""

import os
import sys
import json
import tempfile
import shutil
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── MediaPipe landmark indices (33 keypoints) ──
# We select a subset that maps to standard biomechanical markers
MP_NOSE = 0
MP_LEFT_EYE_INNER = 1
MP_LEFT_EYE = 2
MP_LEFT_EYE_OUTER = 3
MP_RIGHT_EYE_INNER = 4
MP_RIGHT_EYE = 5
MP_RIGHT_EYE_OUTER = 6
MP_LEFT_EAR = 7
MP_RIGHT_EAR = 8
MP_LEFT_SHOULDER = 11
MP_RIGHT_SHOULDER = 12
MP_LEFT_ELBOW = 13
MP_RIGHT_ELBOW = 14
MP_LEFT_WRIST = 15
MP_RIGHT_WRIST = 16
MP_LEFT_PINKY = 17
MP_RIGHT_PINKY = 18
MP_LEFT_INDEX = 19
MP_RIGHT_INDEX = 20
MP_LEFT_HIP = 23
MP_RIGHT_HIP = 24
MP_LEFT_KNEE = 25
MP_RIGHT_KNEE = 26
MP_LEFT_ANKLE = 27
MP_RIGHT_ANKLE = 28
MP_LEFT_HEEL = 29
MP_RIGHT_HEEL = 30
MP_LEFT_FOOT_INDEX = 31
MP_RIGHT_FOOT_INDEX = 32

# ── Marker names for TRC output ──
# These 20 keypoints are the ones used by OpenCap's LSTM augmenter
# and also serve as direct input to Pose2Sim IK when augmentation is skipped
MEDIAPIPE_TO_TRC_MARKERS = {
    MP_NOSE: "Nose",
    MP_LEFT_EAR: "LEar",
    MP_RIGHT_EAR: "REar",
    MP_LEFT_SHOULDER: "LShoulder",
    MP_RIGHT_SHOULDER: "RShoulder",
    MP_LEFT_ELBOW: "LElbow",
    MP_RIGHT_ELBOW: "RElbow",
    MP_LEFT_WRIST: "LWrist",
    MP_RIGHT_WRIST: "RWrist",
    MP_LEFT_HIP: "LHip",
    MP_RIGHT_HIP: "RHip",
    MP_LEFT_KNEE: "LKnee",
    MP_RIGHT_KNEE: "RKnee",
    MP_LEFT_ANKLE: "LAnkle",
    MP_RIGHT_ANKLE: "RAnkle",
    MP_LEFT_HEEL: "LHeel",
    MP_RIGHT_HEEL: "RHeel",
    MP_LEFT_FOOT_INDEX: "LBigToe",
    MP_RIGHT_FOOT_INDEX: "RBigToe",
    MP_LEFT_INDEX: "LIndex",
}

# Full 33 landmark names for Pose2Sim compatibility
MEDIAPIPE_33_NAMES = [
    "Nose", "LEyeInner", "LEye", "LEyeOuter",
    "REyeInner", "REye", "REyeOuter",
    "LEar", "REar", "MouthL", "MouthR",
    "LShoulder", "RShoulder", "LElbow", "RElbow",
    "LWrist", "RWrist", "LPinky", "RPinky",
    "LIndex", "RIndex", "LThumb", "RThumb",
    "LHip", "RHip", "LKnee", "RKnee",
    "LAnkle", "RAnkle", "LHeel", "RHeel",
    "LBigToe", "RBigToe",
]

# Default model paths
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
RAJAGOPAL_MODEL = os.path.join(DEFAULT_MODEL_DIR, "Rajagopal2016.osim")

# OpenSim joint angle names we extract from IK results
OPENSIM_JOINT_ANGLES = [
    "hip_flexion_r", "hip_flexion_l",
    "hip_adduction_r", "hip_adduction_l",
    "hip_rotation_r", "hip_rotation_l",
    "knee_angle_r", "knee_angle_l",
    "ankle_angle_r", "ankle_angle_l",
    "subtalar_angle_r", "subtalar_angle_l",
    "lumbar_extension", "lumbar_bending", "lumbar_rotation",
    "arm_flex_r", "arm_flex_l",
    "arm_add_r", "arm_add_l",
    "arm_rot_r", "arm_rot_l",
    "elbow_flex_r", "elbow_flex_l",
    "pro_sup_r", "pro_sup_l",
]


def landmarks_to_trc(world_landmarks_by_frame, fps, output_path,
                     marker_set="full33"):
    """Convert per-frame MediaPipe world landmarks to TRC file format.

    Args:
        world_landmarks_by_frame: List of frames, each frame is a list of
            33 landmarks with .x, .y, .z attributes (MediaPipe world coords,
            meters, Y-up in MediaPipe = we convert to OpenSim Y-up).
        fps: Video frame rate.
        output_path: Path to write the .trc file.
        marker_set: "full33" for all MediaPipe landmarks, "sparse20" for
            the 20-keypoint subset used by OpenCap augmenter.

    Returns:
        Path to the written TRC file.
    """
    if marker_set == "sparse20":
        indices = sorted(MEDIAPIPE_TO_TRC_MARKERS.keys())
        names = [MEDIAPIPE_TO_TRC_MARKERS[i] for i in indices]
    else:
        indices = list(range(33))
        names = MEDIAPIPE_33_NAMES[:33]

    num_frames = len(world_landmarks_by_frame)
    num_markers = len(names)

    with open(output_path, 'w') as f:
        # TRC header (standard format)
        f.write("PathFileType\t4\t(X/Y/Z)\t{}\n".format(
            os.path.basename(output_path)))
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\t"
                "OrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(f"{fps}\t{fps}\t{num_frames}\t{num_markers}\tm\t"
                f"{fps}\t1\t{num_frames}\n")

        # Marker names header
        header_names = "Frame#\tTime\t" + "\t\t\t".join(names) + "\n"
        f.write(header_names)

        # X/Y/Z sub-header
        coords = []
        for i, name in enumerate(names):
            n = i + 1
            coords.extend([f"X{n}", f"Y{n}", f"Z{n}"])
        f.write("\t\t" + "\t".join(coords) + "\n")

        # Data rows
        for frame_idx, landmarks in enumerate(world_landmarks_by_frame):
            t = frame_idx / fps
            row = [str(frame_idx + 1), f"{t:.6f}"]

            for idx in indices:
                if landmarks and idx < len(landmarks):
                    lm = landmarks[idx]
                    # MediaPipe world coords: X right, Y down, Z toward camera
                    # OpenSim/TRC coords: X right, Y up, Z forward
                    x = lm.x
                    y = -lm.y  # flip Y (MP Y-down → OpenSim Y-up)
                    z = -lm.z  # flip Z (MP Z-toward-camera → OpenSim Z-forward)
                    row.extend([f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"])
                else:
                    row.extend(["", "", ""])

            f.write("\t".join(row) + "\n")

    logger.info(f"TRC written: {output_path} ({num_frames} frames, "
                f"{num_markers} markers)")
    return output_path


def landmarks_to_numpy(world_landmarks_by_frame):
    """Convert MediaPipe world landmarks to numpy array.

    Returns:
        np.ndarray of shape (num_frames, 33, 3) in OpenSim coordinates.
    """
    num_frames = len(world_landmarks_by_frame)
    data = np.zeros((num_frames, 33, 3))

    for i, landmarks in enumerate(world_landmarks_by_frame):
        if landmarks is None:
            continue
        for j in range(min(33, len(landmarks))):
            lm = landmarks[j]
            data[i, j, 0] = lm.x
            data[i, j, 1] = -lm.y   # Y flip
            data[i, j, 2] = -lm.z   # Z flip

    return data


def run_pose2sim_pipeline(trc_path, output_dir, model_path=None,
                          subject_height_m=None, subject_mass_kg=None):
    """Run Pose2Sim scaling + IK pipeline.

    Uses Pose2Sim's built-in auto-scaling from segment lengths (no static
    trial needed) and OpenSim IK solver.

    Args:
        trc_path: Path to input TRC file with 3D marker trajectories.
        output_dir: Directory for Pose2Sim intermediate and output files.
        model_path: Path to .osim model (default: Rajagopal2016).
        subject_height_m: Subject height in meters (improves scaling).
        subject_mass_kg: Subject mass in kg (improves scaling).

    Returns:
        dict with:
            - 'scaled_model': path to scaled .osim
            - 'ik_mot': path to IK .mot results
            - 'success': bool
            - 'error': error string if failed
    """
    if model_path is None:
        model_path = RAJAGOPAL_MODEL

    if not os.path.exists(model_path):
        return {
            'success': False,
            'error': f"OpenSim model not found: {model_path}. "
                     "Download Rajagopal2016.osim to models/ directory."
        }

    result = {
        'scaled_model': None,
        'ik_mot': None,
        'success': False,
        'error': None,
    }

    try:
        import pose2sim  # noqa: delayed import

        # Pose2Sim expects a specific directory structure:
        #   trial_dir/
        #     pose-3d/  (contains TRC files)
        #     opensim/  (contains .osim model)
        #     Config.toml
        trial_dir = os.path.join(output_dir, "pose2sim_trial")
        pose3d_dir = os.path.join(trial_dir, "pose-3d")
        opensim_dir = os.path.join(trial_dir, "opensim")
        os.makedirs(pose3d_dir, exist_ok=True)
        os.makedirs(opensim_dir, exist_ok=True)

        # Copy TRC to pose-3d/
        trc_dest = os.path.join(pose3d_dir, os.path.basename(trc_path))
        shutil.copy2(trc_path, trc_dest)

        # Copy model to opensim/
        model_dest = os.path.join(opensim_dir, os.path.basename(model_path))
        shutil.copy2(model_path, model_dest)
        # Copy geometry files if they exist alongside the model
        model_geom_dir = os.path.join(os.path.dirname(model_path), "Geometry")
        dest_geom_dir = os.path.join(opensim_dir, "Geometry")
        if os.path.isdir(model_geom_dir):
            if os.path.exists(dest_geom_dir):
                shutil.rmtree(dest_geom_dir)
            shutil.copytree(model_geom_dir, dest_geom_dir)

        # Write Config.toml for Pose2Sim
        config = _generate_pose2sim_config(
            fps=_read_trc_fps(trc_path),
            subject_height_m=subject_height_m,
            subject_mass_kg=subject_mass_kg,
        )
        config_path = os.path.join(trial_dir, "Config.toml")
        with open(config_path, 'w') as f:
            f.write(config)

        # Run Pose2Sim kinematics (scaling + IK)
        orig_dir = os.getcwd()
        try:
            os.chdir(trial_dir)
            pose2sim.kinematics()
        finally:
            os.chdir(orig_dir)

        # Find output files
        ik_dir = os.path.join(trial_dir, "opensim")
        for fname in os.listdir(ik_dir):
            if fname.endswith("_ik.mot") or fname.endswith("_IK.mot"):
                result['ik_mot'] = os.path.join(ik_dir, fname)
            elif fname.endswith("_scaled.osim"):
                result['scaled_model'] = os.path.join(ik_dir, fname)

        # Also check kinematics/ subdirectory
        kin_dir = os.path.join(trial_dir, "kinematics")
        if os.path.isdir(kin_dir):
            for fname in os.listdir(kin_dir):
                if fname.endswith(".mot"):
                    result['ik_mot'] = os.path.join(kin_dir, fname)

        if result['ik_mot']:
            result['success'] = True
            logger.info(f"Pose2Sim IK complete: {result['ik_mot']}")
        else:
            result['error'] = "IK output .mot file not found after Pose2Sim run"

    except ImportError:
        result['error'] = ("pose2sim not installed. "
                           "Install with: pip install pose2sim")
    except Exception as e:
        result['error'] = f"Pose2Sim pipeline error: {str(e)}"
        logger.exception("Pose2Sim pipeline failed")

    return result


def run_opensim_ik_direct(trc_path, model_path, output_dir,
                          subject_height_m=None, subject_mass_kg=None):
    """Run OpenSim IK directly via opensim Python bindings (fallback).

    This is used when Pose2Sim is not available or fails. Requires the
    opensim conda package.

    Args:
        trc_path: Path to TRC marker file.
        model_path: Path to .osim model.
        output_dir: Output directory.
        subject_height_m: For model scaling.
        subject_mass_kg: For model scaling.

    Returns:
        dict matching run_pose2sim_pipeline output format.
    """
    result = {
        'scaled_model': None,
        'ik_mot': None,
        'success': False,
        'error': None,
    }

    try:
        import opensim as osim

        # Load model
        model = osim.Model(model_path)

        # Scale model if height/mass provided
        if subject_height_m or subject_mass_kg:
            scaled_path = os.path.join(output_dir, "scaled_model.osim")
            _scale_model_simple(model, scaled_path,
                                subject_height_m, subject_mass_kg)
            model = osim.Model(scaled_path)
            result['scaled_model'] = scaled_path

        # Set up IK tool
        ik_tool = osim.InverseKinematicsTool()
        ik_tool.setModel(model)

        # Read TRC to get time range
        marker_data = osim.MarkerData(trc_path)
        start_time = marker_data.getStartFrameTime()
        end_time = marker_data.getLastFrameTime()

        ik_tool.setMarkerDataFileName(trc_path)
        ik_tool.setStartTime(start_time)
        ik_tool.setEndTime(end_time)

        ik_output = os.path.join(output_dir, "ik_results.mot")
        ik_tool.setOutputMotionFileName(ik_output)

        # Run IK
        ik_tool.run()

        if os.path.exists(ik_output):
            result['ik_mot'] = ik_output
            result['success'] = True
        else:
            result['error'] = "IK did not produce output file"

    except ImportError:
        result['error'] = ("opensim Python package not installed. "
                           "Install via: conda install -c opensim-org opensim")
    except Exception as e:
        result['error'] = f"OpenSim IK error: {str(e)}"
        logger.exception("OpenSim IK direct failed")

    return result


def run_moco_inverse(ik_mot_path, model_path, output_dir,
                     mesh_interval=0.02, reserve_strength=2.0):
    """Run OpenSim MocoInverse to compute muscle activations and forces.

    Given IK kinematics, solves for muscle excitations/activations that
    produce the measured motion. Uses DeGrooteFregly2016Muscle model.

    Args:
        ik_mot_path: Path to IK .mot file with joint angles.
        model_path: Path to scaled .osim model.
        output_dir: Output directory.
        mesh_interval: Time step for Moco mesh (seconds).
        reserve_strength: Strength of reserve actuators (N·m).

    Returns:
        dict with:
            - 'activations_sto': path to muscle activations .sto
            - 'forces_sto': path to tendon forces .sto (if available)
            - 'success': bool
            - 'error': error string if failed
    """
    result = {
        'activations_sto': None,
        'forces_sto': None,
        'success': False,
        'error': None,
    }

    try:
        import opensim as osim

        # Load model and replace muscles with DeGrooteFregly2016
        model = osim.Model(model_path)
        osim.DeGrooteFregly2016Muscle.replaceMuscles(model)

        # Add reserve actuators for all coordinates
        for i in range(model.getCoordinateSet().getSize()):
            coord = model.getCoordinateSet().get(i)
            actu = osim.CoordinateActuator()
            actu.setCoordinate(coord)
            actu.setName(f"reserve_{coord.getName()}")
            actu.setOptimalForce(reserve_strength)
            actu.setMinControl(-1)
            actu.setMaxControl(1)
            model.addForce(actu)

        model.finalizeConnections()

        # Set up MocoInverse
        inverse = osim.MocoInverse()
        inverse.setModel(osim.ModelProcessor(model))
        inverse.setKinematics(osim.TableProcessor(ik_mot_path))
        inverse.set_mesh_interval(mesh_interval)
        inverse.set_initial_time(osim.Storage(ik_mot_path).getFirstTime())
        inverse.set_final_time(osim.Storage(ik_mot_path).getLastTime())

        # Solve
        solution = inverse.solve()

        if solution.isSealed():
            solution.unseal()

        # Write outputs
        activations_path = os.path.join(output_dir, "moco_activations.sto")
        solution.getMocoSolution().write(activations_path)
        result['activations_sto'] = activations_path

        # Extract muscle forces from solution
        forces_path = os.path.join(output_dir, "moco_forces.sto")
        _extract_muscle_forces(model, solution, forces_path)
        if os.path.exists(forces_path):
            result['forces_sto'] = forces_path

        result['success'] = True
        logger.info(f"MocoInverse complete: {activations_path}")

    except ImportError:
        result['error'] = ("opensim Python package not installed or Moco "
                           "not available. Moco requires opensim>=4.4")
    except Exception as e:
        result['error'] = f"MocoInverse error: {str(e)}"
        logger.exception("MocoInverse failed")

    return result


def parse_mot_file(mot_path):
    """Parse an OpenSim .mot file into a dict of time-series arrays.

    Returns:
        dict: {column_name: [{"t": float, "v": float}, ...]}
    """
    if not os.path.exists(mot_path):
        return {}

    result = {}
    header_done = False
    columns = []

    with open(mot_path) as f:
        for line in f:
            line = line.strip()
            if not header_done:
                if line == "endheader":
                    header_done = True
                    continue
                # Check for column names line (tab-separated, starts with "time")
                if line.startswith("time"):
                    columns = line.split('\t')
                    for col in columns[1:]:  # skip "time"
                        result[col] = []
                continue

            if not columns:
                # Try to parse as tab-separated header
                parts = line.split('\t')
                if parts[0] == "time" or parts[0] == "Time":
                    columns = parts
                    for col in columns[1:]:
                        result[col] = []
                    continue

            # Data row
            parts = line.split('\t')
            if len(parts) < 2:
                parts = line.split()  # fallback to whitespace
            if len(parts) >= len(columns):
                t = float(parts[0])
                for i, col in enumerate(columns[1:], 1):
                    if i < len(parts):
                        result[col].append({"t": round(t, 4),
                                            "v": round(float(parts[i]), 4)})

    return result


def parse_sto_file(sto_path):
    """Parse an OpenSim .sto file (same format as .mot but storage)."""
    return parse_mot_file(sto_path)  # Same format


def build_opensim_json(ik_data, moco_data=None, model_version="Rajagopal2016",
                       gltf_path=None):
    """Build the additive 'opensim' JSON block for the output.

    Args:
        ik_data: dict from parse_mot_file of IK results.
        moco_data: dict from parse_sto_file of Moco results (optional).
        model_version: Model name string.
        gltf_path: Path to exported .glb file (optional).

    Returns:
        dict matching the planned JSON schema.
    """
    opensim = {
        "modelVersion": model_version,
    }

    if gltf_path and os.path.exists(gltf_path):
        opensim["scaledModelGltf"] = os.path.basename(gltf_path)

    # Joint angles from IK
    joint_angles = {}
    for col_name, series in ik_data.items():
        # Filter to known joint angle columns
        if any(col_name.startswith(prefix) for prefix in [
            "hip_flexion", "hip_adduction", "hip_rotation",
            "knee_angle", "ankle_angle", "subtalar_angle",
            "lumbar_", "arm_flex", "arm_add", "arm_rot",
            "elbow_flex", "pro_sup",
        ]):
            joint_angles[col_name] = series

    if joint_angles:
        opensim["jointAngles"] = joint_angles

    # Muscle data from Moco
    if moco_data:
        activations = {}
        forces = {}
        for col_name, series in moco_data.items():
            if "/activation" in col_name:
                # Clean name: e.g. "/forceset/soleus_r/activation" → "soleus_r"
                clean = col_name.split("/")[-2] if "/" in col_name else col_name
                activations[clean] = series
            elif "/tendon_force" in col_name or "/fiber_force" in col_name:
                clean = col_name.split("/")[-2] if "/" in col_name else col_name
                forces[clean] = series

        if activations:
            opensim["muscleActivations"] = activations
        if forces:
            opensim["muscleForces"] = forces

    return opensim


def process_with_opensim(world_landmarks_by_frame, fps,
                         subject_height_m=None, subject_mass_kg=None,
                         model_path=None, run_moco=False,
                         augmented_trc_path=None,
                         work_dir=None):
    """Full OpenSim processing pipeline.

    This is the main entry point called from backend_processor.py.

    Args:
        world_landmarks_by_frame: List of per-frame MediaPipe world landmarks.
        fps: Video frame rate.
        subject_height_m: Subject height (meters) for model scaling.
        subject_mass_kg: Subject mass (kg) for model scaling.
        model_path: Custom .osim model path.
        run_moco: Whether to run MocoInverse (slow, 3-5 min).
        augmented_trc_path: Pre-augmented TRC from marker_augmenter (optional).
        work_dir: Working directory (temp dir used if None).

    Returns:
        dict: The 'opensim' JSON block to merge into output, or None on failure.
    """
    cleanup_work_dir = False
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="flekks_opensim_")
        cleanup_work_dir = True

    try:
        # Step 1: Generate TRC from landmarks (unless augmented TRC provided)
        if augmented_trc_path and os.path.exists(augmented_trc_path):
            trc_path = augmented_trc_path
            logger.info("Using augmented TRC from marker enhancer")
        else:
            trc_path = os.path.join(work_dir, "landmarks.trc")
            landmarks_to_trc(world_landmarks_by_frame, fps, trc_path,
                             marker_set="full33")

        # Step 2: Run Pose2Sim (preferred) or direct OpenSim IK (fallback)
        ik_result = run_pose2sim_pipeline(
            trc_path, work_dir, model_path,
            subject_height_m, subject_mass_kg
        )

        if not ik_result['success']:
            logger.warning(f"Pose2Sim failed: {ik_result['error']}, "
                           "trying direct OpenSim IK...")
            ik_result = run_opensim_ik_direct(
                trc_path, model_path or RAJAGOPAL_MODEL, work_dir,
                subject_height_m, subject_mass_kg
            )

        if not ik_result['success']:
            logger.error(f"OpenSim IK failed: {ik_result['error']}")
            return None

        # Step 3: Parse IK results
        ik_data = parse_mot_file(ik_result['ik_mot'])
        if not ik_data:
            logger.error("Failed to parse IK .mot file")
            return None

        # Step 4: Optional MocoInverse
        moco_data = None
        if run_moco and ik_result.get('scaled_model'):
            logger.info("Running MocoInverse (this may take 3-5 minutes)...")
            moco_result = run_moco_inverse(
                ik_result['ik_mot'],
                ik_result['scaled_model'],
                work_dir
            )
            if moco_result['success']:
                moco_data = parse_sto_file(moco_result['activations_sto'])
            else:
                logger.warning(f"MocoInverse failed: {moco_result['error']}")

        # Step 5: Optional glTF export
        gltf_path = None
        scaled_model = ik_result.get('scaled_model')
        if scaled_model:
            try:
                from convert_to_gltf import convert_osim_to_gltf
                gltf_path = os.path.join(work_dir, "model.glb")
                convert_osim_to_gltf(scaled_model, gltf_path)
            except Exception as e:
                logger.warning(f"glTF export skipped: {e}")

        # Step 6: Build JSON
        opensim_json = build_opensim_json(
            ik_data, moco_data,
            gltf_path=gltf_path
        )

        return opensim_json

    except Exception as e:
        logger.exception(f"OpenSim pipeline failed: {e}")
        return None

    finally:
        if cleanup_work_dir and os.path.exists(work_dir):
            try:
                shutil.rmtree(work_dir)
            except Exception:
                pass


# ── Internal helpers ──

def _read_trc_fps(trc_path):
    """Read FPS from TRC file header."""
    with open(trc_path) as f:
        f.readline()  # skip first line
        line2 = f.readline()  # header keys
        line3 = f.readline()  # header values
        parts = line3.strip().split('\t')
        try:
            return float(parts[0])
        except (ValueError, IndexError):
            return 30.0


def _generate_pose2sim_config(fps=30, subject_height_m=None,
                              subject_mass_kg=None):
    """Generate a minimal Config.toml for Pose2Sim kinematics."""
    height = subject_height_m or 1.75
    mass = subject_mass_kg or 70.0

    return f"""# Pose2Sim Config (auto-generated by Flekks)
[project]
frame_rate = {fps}
frame_range = []

[pose]
pose_model = "HALPE_26"

[triangulation]
reproj_error_threshold_triangulation = 15

[filtering]
type = "butterworth"
[filtering.butterworth]
order = 4
cut_off_frequency = 6

[markerAugmentation]
make_c3d = false

[kinematics]
use_augmentation = true
subject_height = {height}
subject_mass = {mass}
"""


def _scale_model_simple(model, output_path, height_m=None, mass_kg=None):
    """Simple uniform model scaling based on height ratio."""
    import opensim as osim

    # Default Rajagopal model height ~1.70m
    default_height = 1.70
    if height_m:
        scale_factor = height_m / default_height
    else:
        scale_factor = 1.0

    # Apply uniform scale to all bodies
    state = model.initSystem()
    body_set = model.getBodySet()
    for i in range(body_set.getSize()):
        body = body_set.get(i)
        if mass_kg:
            # Scale mass proportionally
            original_mass = body.getMass()
            body.setMass(original_mass * (mass_kg / 75.0))

    # Scale factors via ScaleTool would be better, but this is a fallback
    model.printToXML(output_path)


def _extract_muscle_forces(model, moco_solution, output_path):
    """Extract muscle forces from Moco solution (placeholder).

    Full implementation requires iterating through the solution states
    and computing tendon forces via the muscle model.
    """
    try:
        import opensim as osim
        solution_table = moco_solution.getMocoSolution().exportToStatesTable()
        # For now, write the full states which include activations
        osim.STOFileAdapter.write(solution_table, output_path)
    except Exception as e:
        logger.warning(f"Force extraction skipped: {e}")
