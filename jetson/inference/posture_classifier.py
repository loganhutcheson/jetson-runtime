#!/usr/bin/env python3
import json
import math
from collections import Counter, deque
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

KEYPOINT_FEATURES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
]
CLASS_LABELS = ["good", "okay", "bad"]


def angle_between_deg(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> Optional[float]:
    ab = (a[0] - b[0], a[1] - b[1])
    cb = (c[0] - b[0], c[1] - b[1])
    ab_norm = math.hypot(*ab)
    cb_norm = math.hypot(*cb)
    if ab_norm <= 1e-6 or cb_norm <= 1e-6:
        return None
    dot = (ab[0] * cb[0] + ab[1] * cb[1]) / (ab_norm * cb_norm)
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))


def load_jsonl_reports(path: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _std(values: Sequence[float], mean_value: float) -> float:
    if not values:
        return 1.0
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return max(math.sqrt(variance), 1e-6)


def _transform_point(point: Dict[str, float], center: Tuple[float, float],
                     axis_x: Tuple[float, float], axis_y: Tuple[float, float],
                     scale: float) -> Tuple[float, float]:
    dx = point["x"] - center[0]
    dy = point["y"] - center[1]
    return (
        (dx * axis_x[0] + dy * axis_x[1]) / scale,
        (dx * axis_y[0] + dy * axis_y[1]) / scale,
    )


def _head_center(points: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    if not points:
        return None
    return (_mean(point[0] for point in points), _mean(point[1] for point in points))


def extract_posture_features(detection: Dict[str, object]) -> Optional[Dict[str, float]]:
    keypoints = detection.get("keypoints", {})
    bbox = detection.get("bbox_xyxy", [])
    if "left_shoulder" not in keypoints or "right_shoulder" not in keypoints:
        return None

    left_shoulder = keypoints["left_shoulder"]
    right_shoulder = keypoints["right_shoulder"]
    shoulder_dx = right_shoulder["x"] - left_shoulder["x"]
    shoulder_dy = right_shoulder["y"] - left_shoulder["y"]
    shoulder_width = math.hypot(shoulder_dx, shoulder_dy)
    if shoulder_width <= 1e-6:
        return None

    axis_x = (shoulder_dx / shoulder_width, shoulder_dy / shoulder_width)
    axis_y = (-axis_x[1], axis_x[0])
    chest_center = (
        (left_shoulder["x"] + right_shoulder["x"]) * 0.5,
        (left_shoulder["y"] + right_shoulder["y"]) * 0.5,
    )
    bbox_width = max(float(bbox[2] - bbox[0]), 1e-6) if len(bbox) == 4 else shoulder_width
    bbox_height = max(float(bbox[3] - bbox[1]), 1e-6) if len(bbox) == 4 else shoulder_width

    features: Dict[str, float] = {
        "pose_score": float(detection.get("score", 0.0)),
        "present_count": float(len(keypoints)),
        "conf_mean": _mean(point["conf"] for point in keypoints.values()),
        "shoulder_width_bbox_w": shoulder_width / bbox_width,
        "shoulder_width_bbox_h": shoulder_width / bbox_height,
        "shoulder_tilt_deg": math.degrees(math.atan2(shoulder_dy, shoulder_dx)),
        "shoulder_slope_abs": abs(shoulder_dy) / shoulder_width,
    }

    head_points: List[Tuple[float, float]] = []
    transformed_points: Dict[str, Tuple[float, float]] = {}
    for name in KEYPOINT_FEATURES:
        point = keypoints.get(name)
        present = 1.0 if point else 0.0
        conf = float(point["conf"]) if point else 0.0
        x_value = 0.0
        y_value = 0.0
        if point:
            x_value, y_value = _transform_point(point, chest_center, axis_x, axis_y, shoulder_width)
            transformed_points[name] = (x_value, y_value)
            if name in {"nose", "left_eye", "right_eye", "left_ear", "right_ear"}:
                head_points.append((x_value, y_value))
        features[f"{name}_present"] = present
        features[f"{name}_conf"] = conf
        features[f"{name}_x"] = x_value
        features[f"{name}_y"] = y_value

    head_center = _head_center(head_points)
    features["head_center_x"] = head_center[0] if head_center else 0.0
    features["head_center_y"] = head_center[1] if head_center else 0.0
    features["head_present"] = 1.0 if head_center else 0.0

    for side in ("left", "right"):
        shoulder = transformed_points.get(f"{side}_shoulder")
        elbow = transformed_points.get(f"{side}_elbow")
        wrist = transformed_points.get(f"{side}_wrist")
        if shoulder and elbow:
            features[f"{side}_upper_arm_len"] = math.hypot(elbow[0] - shoulder[0], elbow[1] - shoulder[1])
        else:
            features[f"{side}_upper_arm_len"] = 0.0
        if elbow and wrist:
            features[f"{side}_forearm_len"] = math.hypot(wrist[0] - elbow[0], wrist[1] - elbow[1])
        else:
            features[f"{side}_forearm_len"] = 0.0
        elbow_angle = None
        if shoulder and elbow and wrist:
            elbow_angle = angle_between_deg(shoulder, elbow, wrist)
        features[f"{side}_elbow_angle_deg"] = elbow_angle if elbow_angle is not None else 0.0

    if "left_wrist" in transformed_points and "right_wrist" in transformed_points:
        left_wrist = transformed_points["left_wrist"]
        right_wrist = transformed_points["right_wrist"]
        features["wrist_gap_x"] = abs(right_wrist[0] - left_wrist[0])
        features["wrist_gap_y"] = abs(right_wrist[1] - left_wrist[1])
        features["wrist_mid_y"] = (right_wrist[1] + left_wrist[1]) * 0.5
    else:
        features["wrist_gap_x"] = 0.0
        features["wrist_gap_y"] = 0.0
        features["wrist_mid_y"] = 0.0

    if "left_elbow" in transformed_points and "right_elbow" in transformed_points:
        left_elbow = transformed_points["left_elbow"]
        right_elbow = transformed_points["right_elbow"]
        features["elbow_gap_x"] = abs(right_elbow[0] - left_elbow[0])
        features["elbow_gap_y"] = abs(right_elbow[1] - left_elbow[1])
        features["elbow_mid_y"] = (right_elbow[1] + left_elbow[1]) * 0.5
    else:
        features["elbow_gap_x"] = 0.0
        features["elbow_gap_y"] = 0.0
        features["elbow_mid_y"] = 0.0

    return features


def aggregate_feature_rows(rows: Sequence[Dict[str, float]], feature_names: Optional[Sequence[str]] = None) -> Dict[str, float]:
    if not rows:
        raise ValueError("cannot aggregate an empty feature window")
    if feature_names is None:
        feature_names = sorted(rows[0].keys())
    return {
        feature_name: _mean(row.get(feature_name, 0.0) for row in rows)
        for feature_name in feature_names
    }


def build_feature_windows(reports: Sequence[Dict[str, object]], window_size: int,
                          stride: int, min_valid_frames: int) -> List[Dict[str, float]]:
    frame_features: List[Optional[Dict[str, float]]] = []
    for report in reports:
        primary = report.get("primary_detection")
        frame_features.append(extract_posture_features(primary) if primary else None)

    windows: List[Dict[str, float]] = []
    for start in range(0, len(frame_features) - window_size + 1, stride):
        chunk = [row for row in frame_features[start : start + window_size] if row is not None]
        if len(chunk) < min_valid_frames:
            continue
        windows.append(aggregate_feature_rows(chunk))
    return windows


def _vectorize(features: Dict[str, float], feature_names: Sequence[str]) -> List[float]:
    return [float(features.get(name, 0.0)) for name in feature_names]


def _standardize(vector: Sequence[float], means: Sequence[float], stds: Sequence[float]) -> List[float]:
    return [(value - mean) / std for value, mean, std in zip(vector, means, stds)]


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def _softmax(scores: Sequence[float]) -> List[float]:
    if not scores:
        return []
    max_score = max(scores)
    exps = [math.exp(score - max_score) for score in scores]
    total = sum(exps)
    return [value / total for value in exps]


def train_posture_model(label_to_windows: Dict[str, Sequence[Dict[str, float]]],
                        validation_stride: int = 5, knn_k: int = 3) -> Tuple[Dict[str, object], Dict[str, object]]:
    all_feature_names = sorted(
        {name for windows in label_to_windows.values() for row in windows for name in row.keys()}
    )
    if not all_feature_names:
        raise ValueError("no posture features were extracted from the training data")

    all_rows: List[Tuple[str, Dict[str, float]]] = []
    for label, windows in label_to_windows.items():
        for row in windows:
            all_rows.append((label, row))
    if not all_rows:
        raise ValueError("no training windows were produced")

    raw_vectors = [_vectorize(row, all_feature_names) for _, row in all_rows]
    means = [_mean(vector[index] for vector in raw_vectors) for index in range(len(all_feature_names))]
    stds = [_std([vector[index] for vector in raw_vectors], means[index]) for index in range(len(all_feature_names))]

    vectors_by_label: Dict[str, List[List[float]]] = {label: [] for label in label_to_windows}
    examples: List[Dict[str, object]] = []
    for label, row in all_rows:
        vector = _standardize(_vectorize(row, all_feature_names), means, stds)
        vectors_by_label[label].append(vector)
        examples.append({"label": label, "vector": vector})

    model = {
        "version": 1,
        "feature_names": all_feature_names,
        "means": means,
        "stds": stds,
        "labels": list(label_to_windows.keys()),
        "examples": examples,
        "knn_k": max(int(knn_k), 1),
    }

    training_predictions = evaluate_posture_model(model, label_to_windows)
    validation_predictions = evaluate_posture_model(
        model,
        {
            label: [row for idx, row in enumerate(rows) if idx % validation_stride == 0]
            for label, rows in label_to_windows.items()
        },
    )
    report = {
        "training_accuracy": training_predictions["accuracy"],
        "validation_accuracy": validation_predictions["accuracy"],
        "training_confusion": training_predictions["confusion"],
        "validation_confusion": validation_predictions["confusion"],
        "class_counts": {label: len(rows) for label, rows in label_to_windows.items()},
        "feature_count": len(all_feature_names),
        "validation_stride": validation_stride,
        "knn_k": model["knn_k"],
    }
    model["report"] = report
    return model, report


def predict_posture(model: Dict[str, object], features: Dict[str, float]) -> Dict[str, object]:
    feature_names = model["feature_names"]
    means = model["means"]
    stds = model["stds"]
    labels = model["labels"]
    examples = model["examples"]
    knn_k = max(int(model.get("knn_k", 3)), 1)

    vector = _standardize(_vectorize(features, feature_names), means, stds)
    ranked_examples = sorted(
        (
            (_distance(vector, example["vector"]), str(example["label"]))
            for example in examples
        ),
        key=lambda item: item[0],
    )
    votes = {label: 0.0 for label in labels}
    nearest_distances = {label: None for label in labels}
    for distance_value, label in ranked_examples[:knn_k]:
        votes[label] += 1.0 / max(distance_value, 1e-6)
    for distance_value, label in ranked_examples:
        if nearest_distances[label] is None:
            nearest_distances[label] = distance_value
        if all(value is not None for value in nearest_distances.values()):
            break
    probabilities = _softmax([votes[label] for label in labels])
    probs_by_label = {label: probability for label, probability in zip(labels, probabilities)}
    best_label = max(labels, key=lambda label: probs_by_label[label])
    return {
        "label": best_label,
        "confidence": probs_by_label[best_label],
        "probabilities": probs_by_label,
        "votes": votes,
        "nearest_distances": nearest_distances,
    }


def evaluate_posture_model(model: Dict[str, object],
                           label_to_windows: Dict[str, Sequence[Dict[str, float]]]) -> Dict[str, object]:
    confusion: Dict[str, Counter] = {
        label: Counter({predicted: 0 for predicted in model["labels"]})
        for label in model["labels"]
    }
    correct = 0
    total = 0
    for label, windows in label_to_windows.items():
        for row in windows:
            prediction = predict_posture(model, row)
            confusion[label][prediction["label"]] += 1
            total += 1
            if prediction["label"] == label:
                correct += 1
    serializable_confusion = {
        label: dict(confusion[label])
        for label in model["labels"]
    }
    return {
        "accuracy": float(correct / total) if total else 0.0,
        "confusion": serializable_confusion,
    }


def save_posture_model(path: str, model: Dict[str, object]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")


def load_posture_model(path: str) -> Dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class PostureSmoother:
    def __init__(self, labels: Sequence[str], alpha: float) -> None:
        self.labels = list(labels)
        self.alpha = max(0.0, min(alpha, 1.0))
        self.state = {label: 0.0 for label in self.labels}

    def update(self, probabilities: Dict[str, float]) -> Dict[str, float]:
        if not any(self.state.values()):
            self.state = {
                label: float(probabilities.get(label, 0.0))
                for label in self.labels
            }
            return dict(self.state)
        self.state = {
            label: (1.0 - self.alpha) * self.state[label] + self.alpha * float(probabilities.get(label, 0.0))
            for label in self.labels
        }
        total = sum(self.state.values())
        if total > 0.0:
            self.state = {label: value / total for label, value in self.state.items()}
        return dict(self.state)


class PostureWindowBuffer:
    def __init__(self, window_size: int) -> None:
        self.rows: Deque[Dict[str, float]] = deque(maxlen=max(window_size, 1))

    def append(self, row: Optional[Dict[str, float]]) -> None:
        if row is not None:
            self.rows.append(row)

    def ready(self, min_rows: int) -> bool:
        return len(self.rows) >= max(min_rows, 1)

    def aggregate(self, feature_names: Sequence[str]) -> Dict[str, float]:
        if not self.rows:
            raise ValueError("cannot aggregate an empty posture buffer")
        return aggregate_feature_rows(list(self.rows), feature_names)
