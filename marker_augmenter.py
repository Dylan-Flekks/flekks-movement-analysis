#!/usr/bin/env python3
"""
Flekks Marker Augmenter — LSTM-based Keypoint → Anatomical Marker Enhancement

Ports the OpenCap LSTM marker augmentation pipeline for standalone use.
Takes sparse MediaPipe keypoints (20 points) and augments them to 43
dense anatomical markers suitable for high-quality OpenSim IK.

Based on: stanfordnmbl/marker-enhancer and stanfordnmbl/opencap-core

Pipeline:
  MediaPipe 33 landmarks → select 20 keypoints → normalize → LSTM → 43 markers → TRC

The LSTM model was trained on 1433 hours of optical motion capture data
from 1176 subjects (OpenCap v0.3 default). Body markers achieve ~8mm
mean error, arm markers ~15mm.

Usage:
    from marker_augmenter import augment_markers
    augmented_trc = augment_markers(world_landmarks_by_frame, fps, output_trc_path)
"""

import os
import sys
import logging
import numpy as np

logger = logging.getLogger(__name__)

# ── Model paths ──
DEFAULT_WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "models",
                                   "marker_augmenter")

# Pre-trained LSTM model files (download from stanfordnmbl/marker-enhancer)
BODY_MODEL_PATH = os.path.join(DEFAULT_WEIGHTS_DIR, "LSTM_body.h5")
ARMS_MODEL_PATH = os.path.join(DEFAULT_WEIGHTS_DIR, "LSTM_arms.h5")

# Alternative: single combined model
COMBINED_MODEL_PATH = os.path.join(DEFAULT_WEIGHTS_DIR, "LSTM_combined.h5")

# ── MediaPipe → OpenCap keypoint mapping ──
# OpenCap uses 20 sparse keypoints as LSTM input. These map to MediaPipe indices.
# The ordering and naming matches the OpenCap utilsAugmenter.py conventions.

OPENCAP_INPUT_KEYPOINTS = {
    # Name → MediaPipe landmark index
    "Nose": 0,
    "LEar": 7,
    "REar": 8,
    "LShoulder": 11,
    "RShoulder": 12,
    "LElbow": 13,
    "RElbow": 14,
    "LWrist": 15,
    "RWrist": 16,
    "LHip": 23,
    "RHip": 24,
    "LKnee": 25,
    "RKnee": 26,
    "LAnkle": 27,
    "RAnkle": 28,
    "LHeel": 29,
    "RHeel": 30,
    "LBigToe": 31,
    "RBigToe": 32,
    "LIndex": 19,
}

INPUT_KEYPOINT_ORDER = [
    "Nose", "LEar", "REar",
    "LShoulder", "RShoulder",
    "LElbow", "RElbow",
    "LWrist", "RWrist",
    "LHip", "RHip",
    "LKnee", "RKnee",
    "LAnkle", "RAnkle",
    "LHeel", "RHeel",
    "LBigToe", "RBigToe",
    "LIndex",
]

# ── Output: 43 anatomical markers ──
# These are the dense marker set output by the LSTM, matching
# standard optical motion capture marker placements.

# Body markers (35 from body LSTM)
BODY_OUTPUT_MARKERS = [
    "C7", "r_shoulder", "L_shoulder",  # torso top
    "RASI", "LASI", "RPSI", "LPSI",   # pelvis
    "r_ASIS", "L_ASIS",               # ASIS alternate
    "r_knee_study", "L_knee_study",    # knee medial
    "r_mknee_study", "L_mknee_study",  # knee lateral
    "r_ankle_study", "L_ankle_study",  # ankle lateral
    "r_mankle_study", "L_mankle_study", # ankle medial
    "r_calc", "L_calc",               # calcaneus
    "r_toe_study", "L_toe_study",      # toe (2nd met head)
    "r_5meta", "L_5meta",             # 5th metatarsal
    "r_thigh1", "r_thigh2", "r_thigh3",  # right thigh cluster
    "L_thigh1", "L_thigh2", "L_thigh3",  # left thigh cluster
    "r_sh1", "r_sh2", "r_sh3",        # right shank cluster
    "L_sh1", "L_sh2", "L_sh3",        # left shank cluster
    "CLAV",                            # clavicle
]

# Arm markers (8 from arm LSTM)
ARM_OUTPUT_MARKERS = [
    "r_lelbow_study", "L_lelbow_study",   # lateral elbow
    "r_melbow_study", "L_melbow_study",   # medial elbow
    "r_lwrist_study", "L_lwrist_study",   # lateral wrist
    "r_mwrist_study", "L_mwrist_study",   # medial wrist
]

ALL_OUTPUT_MARKERS = BODY_OUTPUT_MARKERS + ARM_OUTPUT_MARKERS


def extract_sparse_keypoints(world_landmarks_by_frame):
    """Extract 20 sparse keypoints from MediaPipe world landmarks.

    Args:
        world_landmarks_by_frame: List of frames, each a list of 33 landmarks.

    Returns:
        np.ndarray of shape (num_frames, 20, 3) in OpenSim coordinates.
    """
    num_frames = len(world_landmarks_by_frame)
    keypoints = np.zeros((num_frames, len(INPUT_KEYPOINT_ORDER), 3))

    for frame_idx, landmarks in enumerate(world_landmarks_by_frame):
        if landmarks is None:
            continue
        for kp_idx, kp_name in enumerate(INPUT_KEYPOINT_ORDER):
            mp_idx = OPENCAP_INPUT_KEYPOINTS[kp_name]
            if mp_idx < len(landmarks):
                lm = landmarks[mp_idx]
                # MediaPipe world → OpenSim coords
                keypoints[frame_idx, kp_idx, 0] = lm.x
                keypoints[frame_idx, kp_idx, 1] = -lm.y  # Y flip
                keypoints[frame_idx, kp_idx, 2] = -lm.z  # Z flip

    return keypoints


def normalize_keypoints(keypoints):
    """Normalize keypoints for LSTM input.

    Centers on hip midpoint and scales by torso length per frame.
    Matches OpenCap's normalization approach.

    Args:
        keypoints: np.ndarray of shape (num_frames, 20, 3).

    Returns:
        (normalized_keypoints, centers, scales) for de-normalization.
    """
    num_frames = keypoints.shape[0]

    # Hip indices in our keypoint order
    lhip_idx = INPUT_KEYPOINT_ORDER.index("LHip")
    rhip_idx = INPUT_KEYPOINT_ORDER.index("RHip")
    lshoulder_idx = INPUT_KEYPOINT_ORDER.index("LShoulder")
    rshoulder_idx = INPUT_KEYPOINT_ORDER.index("RShoulder")

    centers = np.zeros((num_frames, 3))
    scales = np.zeros(num_frames)

    for i in range(num_frames):
        # Center: hip midpoint
        center = (keypoints[i, lhip_idx] + keypoints[i, rhip_idx]) / 2.0
        centers[i] = center

        # Scale: distance from hip midpoint to shoulder midpoint (torso length)
        shoulder_mid = (keypoints[i, lshoulder_idx] +
                        keypoints[i, rshoulder_idx]) / 2.0
        torso_len = np.linalg.norm(shoulder_mid - center)
        scales[i] = torso_len if torso_len > 0.01 else 1.0

    # Normalize
    normalized = keypoints.copy()
    for i in range(num_frames):
        normalized[i] = (keypoints[i] - centers[i]) / scales[i]

    return normalized, centers, scales


def denormalize_markers(markers, centers, scales):
    """Reverse normalization on predicted markers.

    Args:
        markers: np.ndarray of shape (num_frames, num_markers, 3).
        centers: np.ndarray of shape (num_frames, 3).
        scales: np.ndarray of shape (num_frames,).

    Returns:
        np.ndarray of shape (num_frames, num_markers, 3) in world coords.
    """
    denorm = markers.copy()
    num_frames = markers.shape[0]
    for i in range(num_frames):
        denorm[i] = markers[i] * scales[i] + centers[i]
    return denorm


def load_lstm_models(body_path=None, arms_path=None, combined_path=None):
    """Load pre-trained LSTM models for marker augmentation.

    Tries combined model first, then separate body + arms models.

    Returns:
        (body_model, arms_model) or (combined_model, None)
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError(
            "TensorFlow is required for LSTM marker augmentation. "
            "Install with: pip install tensorflow>=2.9"
        )

    body_path = body_path or BODY_MODEL_PATH
    arms_path = arms_path or ARMS_MODEL_PATH
    combined_path = combined_path or COMBINED_MODEL_PATH

    # Try combined model first
    if os.path.exists(combined_path):
        logger.info(f"Loading combined LSTM model: {combined_path}")
        model = tf.keras.models.load_model(combined_path, compile=False)
        return model, None

    # Try separate body + arms
    body_model, arms_model = None, None
    if os.path.exists(body_path):
        logger.info(f"Loading body LSTM model: {body_path}")
        body_model = tf.keras.models.load_model(body_path, compile=False)
    if os.path.exists(arms_path):
        logger.info(f"Loading arms LSTM model: {arms_path}")
        arms_model = tf.keras.models.load_model(arms_path, compile=False)

    if body_model is None:
        raise FileNotFoundError(
            f"LSTM model weights not found. Expected at:\n"
            f"  Body: {body_path}\n"
            f"  Arms: {arms_path}\n"
            f"  Combined: {combined_path}\n"
            f"Download from: https://github.com/stanfordnmbl/marker-enhancer"
        )

    return body_model, arms_model


def predict_markers(keypoints, body_model, arms_model=None,
                    sequence_length=60):
    """Run LSTM inference to predict 43 anatomical markers.

    The LSTM expects sequences of normalized keypoints and predicts
    dense marker positions. We process in overlapping windows.

    Args:
        keypoints: np.ndarray of shape (num_frames, 20, 3), normalized.
        body_model: Loaded Keras model for body markers.
        arms_model: Loaded Keras model for arm markers (optional).
        sequence_length: LSTM input sequence length (frames).

    Returns:
        np.ndarray of shape (num_frames, 43, 3), normalized.
    """
    import tensorflow as tf

    num_frames = keypoints.shape[0]

    # Flatten keypoints: (frames, 20, 3) → (frames, 60)
    flat_input = keypoints.reshape(num_frames, -1)

    # Pad if shorter than sequence_length
    if num_frames < sequence_length:
        pad_len = sequence_length - num_frames
        flat_input = np.pad(flat_input,
                            ((0, pad_len), (0, 0)),
                            mode='edge')
        padded_frames = num_frames + pad_len
    else:
        padded_frames = num_frames

    # Process in windows with 50% overlap
    stride = sequence_length // 2
    num_body = len(BODY_OUTPUT_MARKERS)
    num_arms = len(ARM_OUTPUT_MARKERS)
    num_total = num_body + num_arms

    # Accumulate predictions with weights for overlap averaging
    body_sum = np.zeros((padded_frames, num_body * 3))
    body_count = np.zeros((padded_frames, 1))
    arms_sum = np.zeros((padded_frames, num_arms * 3))
    arms_count = np.zeros((padded_frames, 1))

    for start in range(0, padded_frames - sequence_length + 1, stride):
        end = start + sequence_length
        window = flat_input[start:end]
        window_batch = window[np.newaxis, ...]  # (1, seq_len, 60)

        # Body prediction
        body_pred = body_model.predict(window_batch, verbose=0)
        body_pred = body_pred[0]  # (seq_len, num_body*3)
        body_sum[start:end] += body_pred
        body_count[start:end] += 1

        # Arms prediction
        if arms_model is not None:
            arms_pred = arms_model.predict(window_batch, verbose=0)
            arms_pred = arms_pred[0]
            arms_sum[start:end] += arms_pred
            arms_count[start:end] += 1

    # Handle any remaining frames at the end
    if padded_frames > sequence_length:
        remaining_start = padded_frames - sequence_length
        window = flat_input[remaining_start:padded_frames]
        window_batch = window[np.newaxis, ...]

        body_pred = body_model.predict(window_batch, verbose=0)[0]
        body_sum[remaining_start:padded_frames] += body_pred
        body_count[remaining_start:padded_frames] += 1

        if arms_model is not None:
            arms_pred = arms_model.predict(window_batch, verbose=0)[0]
            arms_sum[remaining_start:padded_frames] += arms_pred
            arms_count[remaining_start:padded_frames] += 1

    # Average overlapping predictions
    body_avg = body_sum / np.maximum(body_count, 1)
    body_markers = body_avg[:num_frames].reshape(num_frames, num_body, 3)

    if arms_model is not None:
        arms_avg = arms_sum / np.maximum(arms_count, 1)
        arms_markers = arms_avg[:num_frames].reshape(num_frames, num_arms, 3)
    else:
        # If no arms model, use combined model output
        # (combined model predicts all 43 at once)
        arms_markers = np.zeros((num_frames, num_arms, 3))

    # If using combined model (arms_model is None), body_model output
    # contains all 43 markers
    if arms_model is None and body_markers.shape[1] == num_total:
        all_markers = body_markers
    else:
        all_markers = np.concatenate([body_markers, arms_markers], axis=1)

    return all_markers


def markers_to_trc(markers, marker_names, fps, output_path):
    """Write predicted markers to TRC file.

    Args:
        markers: np.ndarray of shape (num_frames, num_markers, 3).
        marker_names: List of marker name strings.
        fps: Frame rate.
        output_path: Output TRC file path.

    Returns:
        Path to written TRC file.
    """
    num_frames, num_markers, _ = markers.shape

    with open(output_path, 'w') as f:
        # TRC header
        f.write("PathFileType\t4\t(X/Y/Z)\t{}\n".format(
            os.path.basename(output_path)))
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\t"
                "OrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(f"{fps}\t{fps}\t{num_frames}\t{num_markers}\tm\t"
                f"{fps}\t1\t{num_frames}\n")

        # Marker names
        header = "Frame#\tTime\t" + "\t\t\t".join(marker_names) + "\n"
        f.write(header)

        # X/Y/Z sub-header
        coords = []
        for i in range(num_markers):
            n = i + 1
            coords.extend([f"X{n}", f"Y{n}", f"Z{n}"])
        f.write("\t\t" + "\t".join(coords) + "\n")

        # Data
        for frame_idx in range(num_frames):
            t = frame_idx / fps
            row = [str(frame_idx + 1), f"{t:.6f}"]
            for m in range(num_markers):
                x, y, z = markers[frame_idx, m]
                row.extend([f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"])
            f.write("\t".join(row) + "\n")

    logger.info(f"Augmented TRC written: {output_path} "
                f"({num_frames} frames, {num_markers} markers)")
    return output_path


def augment_markers(world_landmarks_by_frame, fps, output_trc_path,
                    body_model_path=None, arms_model_path=None):
    """Full LSTM marker augmentation pipeline.

    Takes MediaPipe world landmarks, extracts 20 sparse keypoints,
    runs LSTM prediction, and outputs 43 anatomical markers as TRC.

    Args:
        world_landmarks_by_frame: List of per-frame MediaPipe world landmarks.
        fps: Video frame rate.
        output_trc_path: Path to write augmented TRC.
        body_model_path: Custom body LSTM weights path.
        arms_model_path: Custom arms LSTM weights path.

    Returns:
        Path to augmented TRC file, or None if augmentation fails.
    """
    try:
        # Step 1: Extract sparse keypoints
        sparse = extract_sparse_keypoints(world_landmarks_by_frame)
        logger.info(f"Extracted {sparse.shape[0]} frames of sparse keypoints")

        # Step 2: Normalize
        normalized, centers, scales = normalize_keypoints(sparse)

        # Step 3: Load LSTM models
        body_model, arms_model = load_lstm_models(
            body_path=body_model_path,
            arms_path=arms_model_path,
        )

        # Step 4: Predict dense markers
        predicted_normalized = predict_markers(
            normalized, body_model, arms_model
        )

        # Step 5: Denormalize
        predicted_world = denormalize_markers(
            predicted_normalized, centers, scales
        )

        # Step 6: Combine sparse input + dense predictions for full marker set
        # The final TRC includes both the original 20 input keypoints
        # and the 43 predicted anatomical markers
        all_marker_names = INPUT_KEYPOINT_ORDER + ALL_OUTPUT_MARKERS
        all_markers = np.concatenate([sparse, predicted_world], axis=1)

        # Step 7: Write TRC
        return markers_to_trc(
            all_markers, all_marker_names, fps, output_trc_path
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


def download_weights(output_dir=None):
    """Download pre-trained LSTM weights from GitHub.

    Downloads from stanfordnmbl/marker-enhancer releases.
    """
    output_dir = output_dir or DEFAULT_WEIGHTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    print(f"LSTM weights directory: {output_dir}")
    print()
    print("To download pre-trained LSTM weights for marker augmentation:")
    print()
    print("  1. Clone the marker-enhancer repo:")
    print("     git clone https://github.com/stanfordnmbl/marker-enhancer.git")
    print()
    print("  2. Copy the trained model files to:")
    print(f"     {output_dir}/")
    print()
    print("  Expected files:")
    print(f"     {BODY_MODEL_PATH}")
    print(f"     {ARMS_MODEL_PATH}")
    print()
    print("  Or, if using a combined model:")
    print(f"     {COMBINED_MODEL_PATH}")
    print()
    print("  See: https://github.com/stanfordnmbl/marker-enhancer")
    print("       https://github.com/stanfordnmbl/opencap-core")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--download":
        download_weights()
    else:
        print("Flekks LSTM Marker Augmenter")
        print()
        print("Usage:")
        print("  python marker_augmenter.py --download   Show weight download instructions")
        print()
        print("  # Programmatic usage:")
        print("  from marker_augmenter import augment_markers")
        print("  trc = augment_markers(world_landmarks, fps, 'output.trc')")
