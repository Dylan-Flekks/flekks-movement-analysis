#!/usr/bin/env python3
"""
Flekks Marker Augmenter — LSTM-based Keypoint → Anatomical Marker Enhancement

Faithful port of the OpenCap LSTM marker augmentation pipeline (v0.3).
Takes sparse MediaPipe keypoints and augments them to 42 dense anatomical
markers suitable for high-quality OpenSim IK.

Based on: stanfordnmbl/opencap-core utilsAugmenter.py + stanfordnmbl/marker-enhancer

Architecture:
  Two separate LSTM models (body + arms), each with its own input keypoints:
    - Lower body: 15 keypoints → 47 features → LSTM → 34 anatomical markers
    - Upper body:  7 keypoints → 23 features → LSTM →  8 anatomical markers
    - Total: 42 augmented anatomical markers

Preprocessing (matches OpenCap exactly):
  1. Extract keypoints from MediaPipe world landmarks
  2. Subtract reference marker (midHip) from all keypoints
  3. Divide by subject height (meters)
  4. Append subject height and mass as constant features
  5. Standardize with training mean/std (from .npy files)
  6. Reshape to (1, num_frames, num_features) for LSTM

Postprocessing:
  1. Reshape output from (1, num_frames, num_response_features)
  2. Multiply by subject height (reverse step 3)
  3. Add back reference marker position (reverse step 2)
  4. Optional floor offset (align minimum Y to ground plane)

Usage:
    from marker_augmenter import augment_markers
    augmented_trc = augment_markers(
        world_landmarks_by_frame, fps, 'output.trc',
        subject_height_m=1.75, subject_mass_kg=70.0
    )
"""

import copy
import json
import os
import sys
import logging
import numpy as np

logger = logging.getLogger(__name__)

# ── Model directory layout ──
# models/marker_augmenter/v0.3_lower/  (body/lower LSTM)
#   ├── model.json
#   ├── weights.h5
#   ├── metadata.json   {"reference_marker": "midHip"}
#   ├── mean.npy         shape (47,)
#   └── std.npy          shape (47,)
# models/marker_augmenter/v0.3_upper/  (arms/upper LSTM)
#   ├── model.json
#   ├── weights.h5
#   ├── metadata.json   {"reference_marker": "midHip"}
#   ├── mean.npy         shape (23,)
#   └── std.npy          shape (23,)

DEFAULT_AUGMENTER_DIR = os.path.join(os.path.dirname(__file__), "models",
                                     "marker_augmenter")
DEFAULT_LOWER_DIR = os.path.join(DEFAULT_AUGMENTER_DIR, "v0.3_lower")
DEFAULT_UPPER_DIR = os.path.join(DEFAULT_AUGMENTER_DIR, "v0.3_upper")

# ── Lower body model: 15 input keypoints ──
# These names match the OpenPose convention used by OpenCap's training data.
LOWER_FEATURE_MARKERS = [
    "Neck", "RShoulder", "LShoulder", "RHip", "LHip",
    "RKnee", "LKnee", "RAnkle", "LAnkle", "RHeel", "LHeel",
    "RSmallToe", "LSmallToe", "RBigToe", "LBigToe",
]

# ── Upper body model: 7 input keypoints ──
UPPER_FEATURE_MARKERS = [
    "Neck", "RShoulder", "LShoulder", "RElbow", "LElbow",
    "RWrist", "LWrist",
]

# ── Lower body output: 34 response markers ──
LOWER_RESPONSE_MARKERS = [
    "r.ASIS_study", "L.ASIS_study", "r.PSIS_study", "L.PSIS_study",
    "r_knee_study", "r_mknee_study", "r_ankle_study", "r_mankle_study",
    "r_toe_study", "r_5meta_study", "r_calc_study",
    "L_knee_study", "L_mknee_study", "L_ankle_study", "L_mankle_study",
    "L_toe_study", "L_calc_study", "L_5meta_study",
    "r_shoulder_study", "L_shoulder_study", "C7_study",
    "r_thigh1_study", "r_thigh2_study", "r_thigh3_study",
    "L_thigh1_study", "L_thigh2_study", "L_thigh3_study",
    "r_sh1_study", "r_sh2_study", "r_sh3_study",
    "L_sh1_study", "L_sh2_study", "L_sh3_study",
    "RHJC_study", "LHJC_study",
]

# ── Upper body output: 8 response markers ──
UPPER_RESPONSE_MARKERS = [
    "r_lelbow_study", "r_melbow_study", "r_lwrist_study", "r_mwrist_study",
    "L_lelbow_study", "L_melbow_study", "L_lwrist_study", "L_mwrist_study",
]

ALL_RESPONSE_MARKERS = LOWER_RESPONSE_MARKERS + UPPER_RESPONSE_MARKERS

# ── MediaPipe landmark indices ──
# Reference: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
MP_LEFT_SHOULDER = 11
MP_RIGHT_SHOULDER = 12
MP_LEFT_ELBOW = 13
MP_RIGHT_ELBOW = 14
MP_LEFT_WRIST = 15
MP_RIGHT_WRIST = 16
MP_LEFT_HIP = 23
MP_RIGHT_HIP = 24
MP_LEFT_KNEE = 25
MP_RIGHT_KNEE = 26
MP_LEFT_ANKLE = 27
MP_RIGHT_ANKLE = 28
MP_LEFT_HEEL = 29
MP_RIGHT_HEEL = 30
MP_LEFT_FOOT_INDEX = 31   # closest to big toe
MP_RIGHT_FOOT_INDEX = 32  # closest to big toe

# Default anthropometric values when not provided
DEFAULT_HEIGHT_M = 1.75
DEFAULT_MASS_KG = 70.0


def _convert_mediapipe_coords(lm):
    """Convert a single MediaPipe world landmark to OpenSim coordinates.

    MediaPipe: X right, Y down, Z toward camera
    OpenSim:   X right, Y up, Z forward (away from camera)
    """
    return np.array([lm.x, -lm.y, -lm.z])


def _get_mp_point(landmarks, idx):
    """Extract a single MediaPipe landmark as [x, y, z] in OpenSim coords."""
    if idx < len(landmarks):
        return _convert_mediapipe_coords(landmarks[idx])
    return np.array([0.0, 0.0, 0.0])


def _estimate_neck(landmarks):
    """Estimate Neck position (OpenPose keypoint 1) from MediaPipe.

    Neck = midpoint of left and right shoulders.
    """
    l_shoulder = _get_mp_point(landmarks, MP_LEFT_SHOULDER)
    r_shoulder = _get_mp_point(landmarks, MP_RIGHT_SHOULDER)
    return (l_shoulder + r_shoulder) / 2.0


def _estimate_small_toe(landmarks, side="right"):
    """Estimate small toe position from MediaPipe landmarks.

    MediaPipe has foot_index (near big toe) but no small toe marker.
    We estimate the small toe by offsetting laterally from the big toe,
    using the foot's medial-lateral direction derived from heel position.

    The small toe is approximately 40% of foot width lateral to the
    big toe marker, and slightly posterior (toward heel).
    """
    if side == "right":
        big_toe = _get_mp_point(landmarks, MP_RIGHT_FOOT_INDEX)
        heel = _get_mp_point(landmarks, MP_RIGHT_HEEL)
        ankle = _get_mp_point(landmarks, MP_RIGHT_ANKLE)
    else:
        big_toe = _get_mp_point(landmarks, MP_LEFT_FOOT_INDEX)
        heel = _get_mp_point(landmarks, MP_LEFT_HEEL)
        ankle = _get_mp_point(landmarks, MP_LEFT_ANKLE)

    # Foot forward direction (heel to toe)
    foot_fwd = big_toe - heel
    foot_len = np.linalg.norm(foot_fwd)
    if foot_len < 1e-6:
        return big_toe

    foot_fwd_unit = foot_fwd / foot_len

    # Up direction (ankle to knee direction, or just Y-up as fallback)
    up = np.array([0.0, 1.0, 0.0])

    # Lateral direction = cross(forward, up), normalized
    lateral = np.cross(foot_fwd_unit, up)
    lat_len = np.linalg.norm(lateral)
    if lat_len < 1e-6:
        return big_toe
    lateral = lateral / lat_len

    # For right foot, small toe is in +lateral direction
    # For left foot, small toe is in -lateral direction
    sign = 1.0 if side == "right" else -1.0

    # Small toe: lateral offset (~4cm) + slightly posterior (~15% foot length)
    small_toe = big_toe + sign * lateral * 0.04 - foot_fwd_unit * foot_len * 0.15
    return small_toe


def extract_keypoints_for_model(world_landmarks_by_frame, marker_names):
    """Extract keypoints from MediaPipe world landmarks for a specific model.

    Maps each OpenCap/OpenPose marker name to the appropriate MediaPipe
    landmark(s), handling computed markers (Neck, SmallToe).

    Args:
        world_landmarks_by_frame: List of frames, each a list of MediaPipe landmarks.
        marker_names: List of OpenCap marker name strings.

    Returns:
        np.ndarray of shape (num_frames, num_markers, 3) in OpenSim coordinates.
    """
    num_frames = len(world_landmarks_by_frame)
    num_markers = len(marker_names)
    keypoints = np.zeros((num_frames, num_markers, 3))

    # Direct MediaPipe index mapping for most markers
    direct_map = {
        "RShoulder": MP_RIGHT_SHOULDER,
        "LShoulder": MP_LEFT_SHOULDER,
        "RElbow": MP_RIGHT_ELBOW,
        "LElbow": MP_LEFT_ELBOW,
        "RWrist": MP_RIGHT_WRIST,
        "LWrist": MP_LEFT_WRIST,
        "RHip": MP_RIGHT_HIP,
        "LHip": MP_LEFT_HIP,
        "RKnee": MP_RIGHT_KNEE,
        "LKnee": MP_LEFT_KNEE,
        "RAnkle": MP_RIGHT_ANKLE,
        "LAnkle": MP_LEFT_ANKLE,
        "RHeel": MP_RIGHT_HEEL,
        "LHeel": MP_LEFT_HEEL,
        "RBigToe": MP_RIGHT_FOOT_INDEX,
        "LBigToe": MP_LEFT_FOOT_INDEX,
    }

    for frame_idx, landmarks in enumerate(world_landmarks_by_frame):
        if landmarks is None:
            continue

        for m_idx, name in enumerate(marker_names):
            if name == "Neck":
                keypoints[frame_idx, m_idx] = _estimate_neck(landmarks)
            elif name == "RSmallToe":
                keypoints[frame_idx, m_idx] = _estimate_small_toe(
                    landmarks, side="right")
            elif name == "LSmallToe":
                keypoints[frame_idx, m_idx] = _estimate_small_toe(
                    landmarks, side="left")
            elif name in direct_map:
                keypoints[frame_idx, m_idx] = _get_mp_point(
                    landmarks, direct_map[name])
            else:
                logger.warning(f"Unknown marker '{name}' at frame {frame_idx}")

    return keypoints


def compute_reference_marker(world_landmarks_by_frame):
    """Compute the reference marker (midHip) for all frames.

    midHip = midpoint of left hip and right hip, in OpenSim coordinates.

    Returns:
        np.ndarray of shape (num_frames, 3)
    """
    num_frames = len(world_landmarks_by_frame)
    ref = np.zeros((num_frames, 3))

    for i, landmarks in enumerate(world_landmarks_by_frame):
        if landmarks is None:
            continue
        l_hip = _get_mp_point(landmarks, MP_LEFT_HIP)
        r_hip = _get_mp_point(landmarks, MP_RIGHT_HIP)
        ref[i] = (l_hip + r_hip) / 2.0

    return ref


def load_augmenter_model(model_dir):
    """Load a pre-trained LSTM augmenter model from a directory.

    Loads model architecture from model.json and weights from weights.h5,
    matching the OpenCap loading convention.

    Also loads preprocessing stats (mean.npy, std.npy) and metadata.

    Args:
        model_dir: Path to model directory containing model.json, weights.h5,
                   mean.npy, std.npy, metadata.json.

    Returns:
        dict with keys: 'model', 'mean', 'std', 'metadata'
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError(
            "TensorFlow is required for LSTM marker augmentation. "
            "Install with: pip install tensorflow>=2.9"
        )

    # Load model architecture + weights (OpenCap convention)
    model_json_path = os.path.join(model_dir, "model.json")
    weights_path = os.path.join(model_dir, "weights.h5")

    if not os.path.exists(model_json_path):
        raise FileNotFoundError(f"Model architecture not found: {model_json_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    with open(model_json_path, 'r') as f:
        model_json = f.read()

    model = tf.keras.models.model_from_json(model_json)
    model.load_weights(weights_path)
    logger.info(f"Loaded LSTM model from {model_dir}")

    # Load preprocessing statistics
    mean_path = os.path.join(model_dir, "mean.npy")
    std_path = os.path.join(model_dir, "std.npy")
    metadata_path = os.path.join(model_dir, "metadata.json")

    train_mean = None
    train_std = None
    metadata = {}

    if os.path.isfile(mean_path):
        train_mean = np.load(mean_path, allow_pickle=True)
        logger.info(f"Loaded training mean: shape {train_mean.shape}")
    if os.path.isfile(std_path):
        train_std = np.load(std_path, allow_pickle=True)
        logger.info(f"Loaded training std: shape {train_std.shape}")
    if os.path.isfile(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        logger.info(f"Loaded metadata: {metadata}")

    return {
        'model': model,
        'mean': train_mean,
        'std': train_std,
        'metadata': metadata,
    }


def preprocess_for_lstm(keypoints_3d, reference_marker_data, subject_height,
                        subject_mass, train_mean, train_std):
    """Preprocess keypoint data for LSTM inference.

    Follows the exact OpenCap pipeline from utilsAugmenter.py:
      1. Flatten (num_frames, num_markers, 3) → (num_frames, num_markers*3)
      2. Subtract reference marker from each marker's XYZ
      3. Divide by subject height
      4. Append height and mass as constant features
      5. Subtract training mean
      6. Divide by training std
      7. Reshape to (1, num_frames, num_features)

    Args:
        keypoints_3d: np.ndarray shape (num_frames, num_markers, 3)
        reference_marker_data: np.ndarray shape (num_frames, 3) — midHip
        subject_height: float, meters
        subject_mass: float, kg
        train_mean: np.ndarray shape (num_features,) or None
        train_std: np.ndarray shape (num_features,) or None

    Returns:
        np.ndarray shape (1, num_frames, num_features) ready for LSTM
    """
    num_frames, num_markers, _ = keypoints_3d.shape

    # Step 1: Flatten to (num_frames, num_markers * 3)
    flat = keypoints_3d.reshape(num_frames, num_markers * 3)

    # Step 2: Subtract reference marker from each marker's XYZ triplet
    norm1 = np.zeros_like(flat)
    for i in range(0, flat.shape[1], 3):
        norm1[:, i:i+3] = flat[:, i:i+3] - reference_marker_data

    # Step 3: Divide by subject height
    norm2 = copy.deepcopy(norm1)
    norm2 = norm2 / subject_height

    # Step 4: Append height and mass as constant features
    inputs = copy.deepcopy(norm2)
    height_col = subject_height * np.ones((num_frames, 1))
    mass_col = subject_mass * np.ones((num_frames, 1))
    inputs = np.concatenate((inputs, height_col), axis=1)
    inputs = np.concatenate((inputs, mass_col), axis=1)

    # Step 5: Subtract training mean
    if train_mean is not None:
        inputs -= train_mean

    # Step 6: Divide by training std
    if train_std is not None:
        inputs /= train_std

    # Step 7: Reshape for LSTM: (1, num_frames, num_features)
    inputs = np.reshape(inputs, (1, inputs.shape[0], inputs.shape[1]))

    return inputs


def postprocess_lstm_output(outputs, reference_marker_data, subject_height):
    """Postprocess LSTM output back to world coordinates.

    Follows the exact OpenCap pipeline:
      1. Reshape from (1, num_frames, num_features) → (num_frames, num_features)
      2. Multiply by subject height (reverse height normalization)
      3. Add back reference marker position (reverse reference subtraction)

    Args:
        outputs: np.ndarray from model.predict(), shape (1, num_frames, num_features)
        reference_marker_data: np.ndarray shape (num_frames, 3) — midHip
        subject_height: float, meters

    Returns:
        np.ndarray shape (num_frames, num_response_features)
    """
    # Step 1: Remove batch dimension
    out = np.reshape(outputs, (outputs.shape[1], outputs.shape[2]))

    # Step 2: Un-normalize with subject height
    unnorm = out * subject_height

    # Step 3: Un-normalize with reference marker
    unnorm2 = np.zeros_like(unnorm)
    for i in range(0, unnorm.shape[1], 3):
        unnorm2[:, i:i+3] = unnorm[:, i:i+3] + reference_marker_data

    return unnorm2


def run_augmenter_model(model_dict, feature_markers, world_landmarks_by_frame,
                        reference_marker_data, subject_height, subject_mass):
    """Run a single augmenter model (lower or upper) through the full pipeline.

    Args:
        model_dict: dict from load_augmenter_model()
        feature_markers: list of input marker name strings
        world_landmarks_by_frame: raw MediaPipe landmarks
        reference_marker_data: np.ndarray shape (num_frames, 3)
        subject_height: float, meters
        subject_mass: float, kg

    Returns:
        np.ndarray shape (num_frames, num_response_markers * 3) in world coords
    """
    # Extract keypoints for this model's feature markers
    keypoints_3d = extract_keypoints_for_model(
        world_landmarks_by_frame, feature_markers
    )
    logger.info(f"Extracted keypoints: {keypoints_3d.shape} for "
                f"{len(feature_markers)} markers")

    # Preprocess
    inputs = preprocess_for_lstm(
        keypoints_3d, reference_marker_data,
        subject_height, subject_mass,
        model_dict['mean'], model_dict['std']
    )
    logger.info(f"Preprocessed input shape: {inputs.shape}")

    # Run LSTM inference
    outputs = model_dict['model'].predict(inputs, verbose=0)
    logger.info(f"LSTM output shape: {outputs.shape}")

    # Postprocess
    result = postprocess_lstm_output(
        outputs, reference_marker_data, subject_height
    )
    logger.info(f"Postprocessed output shape: {result.shape}")

    return result


def markers_to_trc(marker_data_dict, fps, output_path, floor_offset=True):
    """Write augmented markers to a TRC file.

    Args:
        marker_data_dict: dict mapping marker_name → np.ndarray shape (num_frames, 3)
        fps: Frame rate.
        output_path: Output TRC file path.
        floor_offset: If True, offset Y so minimum foot marker is at Y=0.01.

    Returns:
        Path to written TRC file.
    """
    marker_names = list(marker_data_dict.keys())
    num_markers = len(marker_names)
    if num_markers == 0:
        return None

    # Stack all marker data: (num_frames, num_markers, 3)
    first_key = marker_names[0]
    num_frames = marker_data_dict[first_key].shape[0]
    all_data = np.zeros((num_frames, num_markers, 3))
    for i, name in enumerate(marker_names):
        all_data[:, i, :] = marker_data_dict[name]

    # Optional floor offset: align minimum Y to ground
    if floor_offset:
        min_y = np.min(all_data[:, :, 1])  # Y is up in OpenSim coords
        offset_val = -(min_y - 0.01)
        all_data[:, :, 1] += offset_val
        logger.info(f"Applied floor offset: {offset_val:.4f}m "
                    f"(min_y was {min_y:.4f}m)")

    with open(output_path, 'w') as f:
        # TRC header line 1
        f.write("PathFileType\t4\t(X/Y/Z)\t{}\n".format(
            os.path.basename(output_path)))
        # TRC header line 2
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\t"
                "OrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(f"{fps}\t{fps}\t{num_frames}\t{num_markers}\tm\t"
                f"{fps}\t1\t{num_frames}\n")

        # Marker names row
        header = "Frame#\tTime\t" + "\t\t\t".join(marker_names) + "\n"
        f.write(header)

        # X/Y/Z sub-header
        coords = []
        for i in range(num_markers):
            n = i + 1
            coords.extend([f"X{n}", f"Y{n}", f"Z{n}"])
        f.write("\t\t" + "\t".join(coords) + "\n")

        # Blank line (TRC convention)
        f.write("\n")

        # Data rows
        for frame_idx in range(num_frames):
            t = frame_idx / fps
            row = [str(frame_idx + 1), f"{t:.6f}"]
            for m in range(num_markers):
                x, y, z = all_data[frame_idx, m]
                row.extend([f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"])
            f.write("\t".join(row) + "\n")

    logger.info(f"Augmented TRC written: {output_path} "
                f"({num_frames} frames, {num_markers} markers)")
    return output_path


def augment_markers(world_landmarks_by_frame, fps, output_trc_path,
                    subject_height_m=None, subject_mass_kg=None,
                    lower_model_dir=None, upper_model_dir=None,
                    include_input_markers=True, floor_offset=True):
    """Full LSTM marker augmentation pipeline.

    Takes MediaPipe world landmarks, runs two separate LSTM models
    (lower body: 15 keypoints → 34 markers, upper body: 7 keypoints → 8 markers),
    and outputs 42 augmented anatomical markers as a TRC file.

    Optionally includes the original sparse input keypoints in the TRC
    for a total of 42 + input markers.

    Args:
        world_landmarks_by_frame: List of per-frame MediaPipe world landmarks.
        fps: Video frame rate.
        output_trc_path: Path to write augmented TRC.
        subject_height_m: Subject height in meters (default 1.75).
        subject_mass_kg: Subject mass in kg (default 70.0).
        lower_model_dir: Custom path to lower body LSTM model directory.
        upper_model_dir: Custom path to upper body LSTM model directory.
        include_input_markers: If True, include sparse input keypoints in TRC.
        floor_offset: If True, offset Y to align with ground plane.

    Returns:
        Path to augmented TRC file, or None if augmentation fails.
    """
    subject_height = subject_height_m or DEFAULT_HEIGHT_M
    subject_mass = subject_mass_kg or DEFAULT_MASS_KG
    lower_dir = lower_model_dir or DEFAULT_LOWER_DIR
    upper_dir = upper_model_dir or DEFAULT_UPPER_DIR

    try:
        num_frames = len(world_landmarks_by_frame)
        logger.info(f"Starting marker augmentation: {num_frames} frames, "
                    f"height={subject_height}m, mass={subject_mass}kg")

        # Compute reference marker (midHip) for all frames
        ref_marker = compute_reference_marker(world_landmarks_by_frame)

        # Collect all marker data: name → (num_frames, 3)
        marker_data = {}

        # ── Run lower body model ──
        if os.path.isdir(lower_dir):
            logger.info(f"Loading lower body model from {lower_dir}")
            lower_model = load_augmenter_model(lower_dir)

            lower_output = run_augmenter_model(
                lower_model, LOWER_FEATURE_MARKERS,
                world_landmarks_by_frame, ref_marker,
                subject_height, subject_mass
            )

            # Unpack flat output into per-marker arrays
            for c, name in enumerate(LOWER_RESPONSE_MARKERS):
                marker_data[name] = lower_output[:, c*3:(c+1)*3]

            logger.info(f"Lower body: {len(LOWER_RESPONSE_MARKERS)} markers augmented")
        else:
            logger.warning(f"Lower body model dir not found: {lower_dir}")

        # ── Run upper body model ──
        if os.path.isdir(upper_dir):
            logger.info(f"Loading upper body model from {upper_dir}")
            upper_model = load_augmenter_model(upper_dir)

            upper_output = run_augmenter_model(
                upper_model, UPPER_FEATURE_MARKERS,
                world_landmarks_by_frame, ref_marker,
                subject_height, subject_mass
            )

            for c, name in enumerate(UPPER_RESPONSE_MARKERS):
                marker_data[name] = upper_output[:, c*3:(c+1)*3]

            logger.info(f"Upper body: {len(UPPER_RESPONSE_MARKERS)} markers augmented")
        else:
            logger.warning(f"Upper body model dir not found: {upper_dir}")

        if not marker_data:
            logger.error("No models loaded — cannot augment markers")
            return None

        # ── Optionally include sparse input keypoints ──
        if include_input_markers:
            # Use the union of lower + upper feature markers (deduplicated)
            all_input_names = []
            seen = set()
            for name in LOWER_FEATURE_MARKERS + UPPER_FEATURE_MARKERS:
                if name not in seen:
                    all_input_names.append(name)
                    seen.add(name)

            input_kps = extract_keypoints_for_model(
                world_landmarks_by_frame, all_input_names
            )
            for i, name in enumerate(all_input_names):
                # Prefix to distinguish from response markers
                marker_data[f"input_{name}"] = input_kps[:, i, :]

        # ── Write TRC ──
        return markers_to_trc(
            marker_data, fps, output_trc_path, floor_offset=floor_offset
        )

    except ImportError as e:
        logger.warning(f"Marker augmentation unavailable: {e}")
        return None
    except FileNotFoundError as e:
        logger.warning(f"LSTM weights not found: {e}")
        return None
    except Exception as e:
        logger.exception(f"Marker augmentation failed: {e}")
        return None


def verify_models(lower_dir=None, upper_dir=None):
    """Check that model files exist and report status."""
    lower_dir = lower_dir or DEFAULT_LOWER_DIR
    upper_dir = upper_dir or DEFAULT_UPPER_DIR

    required_files = ["model.json", "weights.h5", "mean.npy", "std.npy",
                      "metadata.json"]

    print(f"Marker Augmenter Model Check")
    print(f"{'='*50}")

    for label, model_dir in [("Lower body", lower_dir), ("Upper body", upper_dir)]:
        print(f"\n{label}: {model_dir}")
        if not os.path.isdir(model_dir):
            print(f"  [MISSING] Directory does not exist")
            continue
        for fname in required_files:
            fpath = os.path.join(model_dir, fname)
            if os.path.isfile(fpath):
                size = os.path.getsize(fpath)
                print(f"  [OK] {fname} ({size:,} bytes)")
            else:
                print(f"  [MISSING] {fname}")

    # Check TensorFlow availability
    print(f"\nDependencies:")
    try:
        import tensorflow as tf
        print(f"  [OK] TensorFlow {tf.__version__}")
    except ImportError:
        print(f"  [MISSING] TensorFlow (pip install tensorflow>=2.9)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--verify":
        verify_models()
    else:
        print("Flekks LSTM Marker Augmenter (OpenCap v0.3 port)")
        print()
        print("Usage:")
        print("  python marker_augmenter.py --verify   Check model files")
        print()
        print("  # Programmatic usage:")
        print("  from marker_augmenter import augment_markers")
        print("  trc = augment_markers(")
        print("      world_landmarks, fps, 'output.trc',")
        print("      subject_height_m=1.75, subject_mass_kg=70.0")
        print("  )")
        print()
        print("Models:")
        print(f"  Lower body (15→34): {DEFAULT_LOWER_DIR}")
        print(f"  Upper body (7→8):   {DEFAULT_UPPER_DIR}")
