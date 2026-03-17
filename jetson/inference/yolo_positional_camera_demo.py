#!/usr/bin/env python3
import argparse
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

MODEL_PATH = "/home/logan/models/yolov4-tiny.weights"
MODEL_CONFIG_PATH = "/home/logan/models/yolov4-tiny.cfg"
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


def ensure_model(path: str, config_path: Optional[str]) -> str:
    model_dir = os.path.dirname(path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    if path.endswith(".weights"):
        if not os.path.isfile(path):
            raise SystemExit(
                f"missing {path}.\n"
                "Download the default YOLOv4-tiny weights with:\n"
                f"  wget -O {path} "
                "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights"
            )
        if not config_path:
            raise SystemExit("--config is required when --model points to Darknet weights")
        if not os.path.isfile(config_path):
            raise SystemExit(
                f"missing {config_path}.\n"
                "Download the default YOLOv4-tiny config with:\n"
                f"  wget -O {config_path} "
                "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg"
            )
        return "darknet"

    if os.path.isfile(path):
        return "onnx"

    raise SystemExit(
        f"missing {path}.\n"
        "For ONNX models, place a YOLO ONNX file there first.\n"
        "For the default Jetson path, use the bundled Darknet flow instead:\n"
        f"  wget -O {MODEL_CONFIG_PATH} "
        "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg\n"
        f"  wget -O {MODEL_PATH} "
        "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights"
    )


def csi_pipeline(width: int, height: int, fps: int, sensor_id: int) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, "
        f"framerate=(fraction){fps}/1, format=(string)NV12 ! "
        "nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=true max-buffers=1 sync=false"
    )


def letterbox(frame: np.ndarray, size: int) -> Tuple[np.ndarray, float, float, float]:
    h, w = frame.shape[:2]
    scale = min(size / w, size / h)
    resized_w = int(round(w * scale))
    resized_h = int(round(h * scale))

    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)

    pad_x = (size - resized_w) / 2.0
    pad_y = (size - resized_h) / 2.0
    x0 = int(round(pad_x))
    y0 = int(round(pad_y))
    canvas[y0 : y0 + resized_h, x0 : x0 + resized_w] = resized
    return canvas, scale, pad_x, pad_y


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(v, hi))


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def positional_encoding(x1: float, y1: float, x2: float, y2: float, frame_w: int,
                        frame_h: int, score: float) -> Dict[str, object]:
    cx = ((x1 + x2) * 0.5) / frame_w
    cy = ((y1 + y2) * 0.5) / frame_h
    w = (x2 - x1) / frame_w
    h = (y2 - y1) / frame_h
    area = w * h
    vector = [
        round(cx, 6),
        round(cy, 6),
        round(w, 6),
        round(h, 6),
        round(area, 6),
        round(float(np.sin(np.pi * cx)), 6),
        round(float(np.cos(np.pi * cx)), 6),
        round(float(np.sin(np.pi * cy)), 6),
        round(float(np.cos(np.pi * cy)), 6),
        round(score, 6),
    ]
    return {
        "center_xy_norm": [round(cx, 6), round(cy, 6)],
        "size_wh_norm": [round(w, 6), round(h, 6)],
        "area_norm": round(area, 6),
        "vector": vector,
    }


def bbox_iou(box_a: List[float], box_b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter_area
    return inter_area / denom if denom > 0.0 else 0.0


def attach_tracks(detections: List[Dict[str, object]], prev_tracks: Dict[int, Dict[str, object]],
                  next_track_id: int, match_iou: float) -> Tuple[List[Dict[str, object]],
                                                                 Dict[int, Dict[str, object]], int]:
    next_tracks: Dict[int, Dict[str, object]] = {}
    used_track_ids = set()

    for det in detections:
        box = det["bbox_xyxy"]
        pos = det["position"]
        center = pos["center_xy_norm"]
        best_track_id = None
        best_iou = 0.0

        for track_id, prev in prev_tracks.items():
            if track_id in used_track_ids:
                continue
            iou = bbox_iou(box, prev["bbox_xyxy"])
            if iou >= match_iou and iou > best_iou:
                best_iou = iou
                best_track_id = track_id

        if best_track_id is None:
            track_id = next_track_id
            next_track_id += 1
            dx = 0.0
            dy = 0.0
            darea = 0.0
            age = 1
            matched_iou = 0.0
        else:
            track_id = best_track_id
            used_track_ids.add(track_id)
            prev = prev_tracks[track_id]
            prev_center = prev["center_xy_norm"]
            dx = center[0] - prev_center[0]
            dy = center[1] - prev_center[1]
            darea = pos["area_norm"] - prev["area_norm"]
            age = int(prev["age"]) + 1
            matched_iou = best_iou

        det["track_id"] = track_id
        det["motion"] = {
            "delta_center_xy_norm": [round(dx, 6), round(dy, 6)],
            "delta_area_norm": round(darea, 6),
            "matched_iou": round(matched_iou, 6),
            "speed_norm": round(float(np.sqrt(dx * dx + dy * dy)), 6),
            "age_frames": age,
        }
        next_tracks[track_id] = {
            "bbox_xyxy": box,
            "center_xy_norm": center,
            "area_norm": pos["area_norm"],
            "age": age,
        }

    return detections, next_tracks, next_track_id


def save_detection_crops(frame: np.ndarray, detections: List[Dict[str, object]], out_dir: str,
                         frame_idx: int, min_score: float, every_n: int) -> int:
    saved = 0
    os.makedirs(out_dir, exist_ok=True)

    for det in detections:
        if det["score"] < min_score:
            continue
        if every_n > 1 and frame_idx % every_n != 0:
            continue

        x1, y1, x2, y2 = [int(round(v)) for v in det["bbox_xyxy"]]
        if x2 <= x1 or y2 <= y1:
            continue

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        track_id = det.get("track_id", -1)
        out_path = os.path.join(
            out_dir,
            f"frame_{frame_idx:06d}_track_{track_id:03d}_score_{det['score']:.3f}.jpg",
        )
        if cv2.imwrite(out_path, crop):
            saved += 1

    return saved


def decode_detections(output: np.ndarray, frame_shape: Tuple[int, int], scale: float,
                      pad_x: float, pad_y: float, conf_thres: float, nms_thres: float,
                      person_only: bool) -> List[Dict[str, object]]:
    if output.ndim == 3:
        output = np.squeeze(output, axis=0)
    if output.ndim != 2:
        raise ValueError(f"unexpected YOLO output shape: {output.shape}")
    if output.shape[0] < output.shape[1]:
        output = output.transpose()

    frame_h, frame_w = frame_shape[:2]
    boxes: List[List[int]] = []
    scores: List[float] = []
    detections: List[Dict[str, object]] = []

    for row in output:
        if row.shape[0] < 5:
            continue

        cx, cy, bw, bh = row[:4]
        if row.shape[0] > 84:
            objectness = float(row[4])
            class_scores = row[5:]
        else:
            objectness = 1.0
            class_scores = row[4:]

        class_id = int(np.argmax(class_scores))
        if person_only and class_id != 0:
            continue

        score = objectness * float(class_scores[class_id])
        if score < conf_thres:
            continue

        x1 = (float(cx) - float(bw) * 0.5 - pad_x) / scale
        y1 = (float(cy) - float(bh) * 0.5 - pad_y) / scale
        x2 = (float(cx) + float(bw) * 0.5 - pad_x) / scale
        y2 = (float(cy) + float(bh) * 0.5 - pad_y) / scale
        x1 = clamp(x1, 0.0, frame_w - 1.0)
        y1 = clamp(y1, 0.0, frame_h - 1.0)
        x2 = clamp(x2, 0.0, frame_w - 1.0)
        y2 = clamp(y2, 0.0, frame_h - 1.0)
        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append([int(round(x1)), int(round(y1)), int(round(x2 - x1)), int(round(y2 - y1))])
        scores.append(score)
        detections.append(
            {
                "class_id": class_id,
                "label": COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else str(class_id),
                "score": round(score, 6),
                "bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "position": positional_encoding(x1, y1, x2, y2, frame_w, frame_h, score),
            }
        )

    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, nms_thres)
    if len(indices) == 0:
        return []

    keep = []
    for idx in indices:
        keep.append(int(idx[0] if isinstance(idx, (list, tuple, np.ndarray)) else idx))
    keep.sort(key=lambda idx: detections[idx]["score"], reverse=True)
    return [detections[idx] for idx in keep]


def decode_darknet(outputs: List[np.ndarray], frame_shape: Tuple[int, int], conf_thres: float,
                   nms_thres: float, person_only: bool) -> List[Dict[str, object]]:
    frame_h, frame_w = frame_shape[:2]
    boxes: List[List[int]] = []
    scores: List[float] = []
    detections: List[Dict[str, object]] = []

    for output in outputs:
        for row in output:
            scores_row = row[5:]
            class_id = int(np.argmax(scores_row))
            if person_only and class_id != 0:
                continue

            score = float(scores_row[class_id]) * float(row[4])
            if score < conf_thres:
                continue

            cx = float(row[0]) * frame_w
            cy = float(row[1]) * frame_h
            bw = float(row[2]) * frame_w
            bh = float(row[3]) * frame_h
            x1 = clamp(cx - bw * 0.5, 0.0, frame_w - 1.0)
            y1 = clamp(cy - bh * 0.5, 0.0, frame_h - 1.0)
            x2 = clamp(cx + bw * 0.5, 0.0, frame_w - 1.0)
            y2 = clamp(cy + bh * 0.5, 0.0, frame_h - 1.0)
            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([int(round(x1)), int(round(y1)), int(round(x2 - x1)), int(round(y2 - y1))])
            scores.append(score)
            detections.append(
                {
                    "class_id": class_id,
                    "label": COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else str(class_id),
                    "score": round(score, 6),
                    "bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                    "position": positional_encoding(x1, y1, x2, y2, frame_w, frame_h, score),
                }
            )

    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, nms_thres)
    if len(indices) == 0:
        return []

    keep = []
    for idx in indices:
        keep.append(int(idx[0] if isinstance(idx, (list, tuple, np.ndarray)) else idx))
    keep.sort(key=lambda idx: detections[idx]["score"], reverse=True)
    return [detections[idx] for idx in keep]


def annotate_frame(frame: np.ndarray, detections: List[Dict[str, object]]) -> None:
    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = [int(round(v)) for v in det["bbox_xyxy"]]
        pos = det["position"]
        cx, cy = pos["center_xy_norm"]
        track_id = det.get("track_id", -1)
        motion = det.get("motion", {})
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(
            frame,
            f"{det['label']}#{track_id} {det['score']:.2f}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"pos#{idx} cx={cx:.3f} cy={cy:.3f} dx={motion.get('delta_center_xy_norm', [0.0, 0.0])[0]:.3f}",
            (x1, min(frame.shape[0] - 10, y2 + 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (30, 200, 255),
            2,
            cv2.LINE_AA,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--config", default=MODEL_CONFIG_PATH)
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--sensor-id", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--conf-thres", type=float, default=0.35)
    parser.add_argument("--nms-thres", type=float, default=0.45)
    parser.add_argument("--print-every", type=int, default=15)
    parser.add_argument("--output", default="/home/logan/yolo-positional-demo.avi")
    parser.add_argument("--positions-out", default="/home/logan/yolo-positional-detections.jsonl")
    parser.add_argument("--save-crops-dir", default="")
    parser.add_argument("--save-crops-min-score", type=float, default=0.55)
    parser.add_argument("--save-crops-every", type=int, default=10)
    parser.add_argument("--track-match-iou", type=float, default=0.3)
    parser.add_argument("--all-classes", action="store_true")
    args = parser.parse_args()

    model_kind = ensure_model(args.model, args.config)

    cap = cv2.VideoCapture(
        csi_pipeline(args.width, args.height, args.fps, args.sensor_id),
        cv2.CAP_GSTREAMER,
    )
    if not cap.isOpened():
        raise SystemExit("failed to open CSI camera through GStreamer")

    if model_kind == "darknet":
        net = cv2.dnn.readNetFromDarknet(args.config, args.model)
        darknet_layer_names = net.getUnconnectedOutLayersNames()
    else:
        net = cv2.dnn.readNetFromONNX(args.model)
        darknet_layer_names = []

    ensure_parent_dir(args.output)
    ensure_parent_dir(args.positions_out)
    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"MJPG"),
        min(args.fps, 10),
        (args.width, args.height),
    )

    start = time.time()
    last_report: Dict[str, object] = {}
    prev_tracks: Dict[int, Dict[str, object]] = {}
    next_track_id = 0
    total_saved_crops = 0

    with open(args.positions_out, "w", encoding="utf-8") as positions_file:
        for frame_idx in range(args.frames):
            ok, frame = cap.read()
            if not ok:
                print("frame read failed")
                break
            raw_frame = frame.copy()

            if model_kind == "darknet":
                blob = cv2.dnn.blobFromImage(
                    frame,
                    scalefactor=1.0 / 255.0,
                    size=(args.imgsz, args.imgsz),
                    mean=(0, 0, 0),
                    swapRB=True,
                    crop=False,
                )
                net.setInput(blob)
                detections = decode_darknet(
                    net.forward(darknet_layer_names),
                    frame.shape,
                    args.conf_thres,
                    args.nms_thres,
                    person_only=not args.all_classes,
                )
            else:
                model_input, scale, pad_x, pad_y = letterbox(frame, args.imgsz)
                blob = cv2.dnn.blobFromImage(
                    model_input,
                    scalefactor=1.0 / 255.0,
                    size=(args.imgsz, args.imgsz),
                    mean=(0, 0, 0),
                    swapRB=True,
                    crop=False,
                )
                net.setInput(blob)
                detections = decode_detections(
                    net.forward(),
                    frame.shape,
                    scale,
                    pad_x,
                    pad_y,
                    args.conf_thres,
                    args.nms_thres,
                    person_only=not args.all_classes,
                )

            detections, prev_tracks, next_track_id = attach_tracks(
                detections,
                prev_tracks,
                next_track_id,
                args.track_match_iou,
            )
            annotate_frame(frame, detections)
            writer.write(frame)

            if args.save_crops_dir:
                total_saved_crops += save_detection_crops(
                    raw_frame,
                    detections,
                    args.save_crops_dir,
                    frame_idx,
                    args.save_crops_min_score,
                    args.save_crops_every,
                )

            primary_detection = detections[0] if detections else None

            report = {
                "frame": frame_idx,
                "t_wall_s": round(time.time() - start, 6),
                "frame_size": {
                    "width": int(frame.shape[1]),
                    "height": int(frame.shape[0]),
                },
                "primary_detection": primary_detection,
                "detections": detections,
            }
            positions_file.write(json.dumps(report) + "\n")
            last_report = report

            if frame_idx % args.print_every == 0:
                print(
                    f"frame={frame_idx} persons={len(detections)} "
                    f"primary={primary_detection} saved_crops={total_saved_crops}"
                )

    elapsed = time.time() - start
    cap.release()
    writer.release()

    print(
        f"done frames={last_report.get('frame', -1) + 1} elapsed={elapsed:.2f}s "
        f"fps={max(last_report.get('frame', -1) + 1, 0) / max(elapsed, 1e-6):.2f} "
        f"out={args.output} positions={args.positions_out} saved_crops={total_saved_crops}"
    )
    print(f"last_report={last_report}")


if __name__ == "__main__":
    main()
