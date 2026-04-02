# Posture Classifier

Train a user-specific `good | okay | bad` posture model from labeled JSONL pose captures:

```bash
python3 jetson/inference/train_posture_classifier.py \
  --good /tmp/pose_good.jsonl \
  --okay /tmp/pose_okay.jsonl \
  --bad /tmp/pose_bad.jsonl \
  --model-out /tmp/posture_classifier.json \
  --report-out /tmp/posture_classifier_report.json
```

Run live camera inference with posture classification overlaid into both the AVI and JSONL output:

```bash
python3 jetson/inference/pose_camera_demo.py \
  --backend ultralytics \
  --model /home/logan/models/yolo11n-pose.pt \
  --device 0 \
  --sensor-id 0 \
  --frames 1800 \
  --fps 30 \
  --width 1280 \
  --height 720 \
  --posture-model /tmp/posture_classifier.json \
  --output /tmp/posture_live.avi \
  --pose-out /tmp/posture_live.jsonl
```

Notes:

- The classifier is feature-based, not pixel-based. It uses shoulder-aligned pose geometry and confidence features extracted from the detected skeleton.
- The default runtime smooths predictions across a 20-frame window and writes the smoothed class probabilities into each JSONL record as `posture_prediction`.
- If you want a live preview window on a local Jetson session, add `--show`.
