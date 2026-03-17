----- Jetson Camera Inference Notes -----
This directory now documents the YOLO-based camera inference path we want to
carry forward on the Jetson Orin Nano after the CAM1 IMX219 bring-up.

Date updated:
- 2026-03-16

Hardware / software context:
- Jetson Orin Nano dev kit
- Ubuntu 22.04.5 LTS
- L4T `36.5.0`
- TensorRT `10.3.0`
- CSI camera on `csi://0`

----- What Works -----
Working demo:
- live CSI camera capture through Argus / GStreamer
- frame-by-frame YOLO detection on the CSI stream
- per-detection positional encoding written alongside the video
- lightweight track IDs and frame-to-frame motion deltas for each detection
- annotated output video saved on the Jetson

Primary output paths:
- `/home/logan/yolo-positional-demo.avi`
- `/home/logan/yolo-positional-detections.jsonl`

Working repo script:
- `jetson/inference/yolo_positional_camera_demo.py`
- `jetson/inference/pose_camera_demo.py`

Default model path:
- `yolov4-tiny` via Darknet weights + cfg because it is easy to fetch on this Jetson

Optional model path:
- YOLO-family ONNX is also supported by the same script

Pose overlay path:
- a separate YOLO pose ONNX demo is available for real-time landmarks and spine proxy visualization

What this proves:
- camera capture is healthy
- Python OpenCV can read the CSI stream
- a YOLO-family detector can run against live frames
- results can be rendered and saved for review
- each detection can be turned into a compact position vector for downstream use
- person crops can be exported for later fine-tuning if the base detector underperforms

What it does not prove:
- multi-person tracking identity over time
- posture classification
- training on our custom data

----- Why This Path Was Needed -----
We also rebuilt `jetson-inference` on the Jetson and got its native C++ binaries
to compile on this JetPack. However, the stock `jetson-inference` example models
were not the quickest path to a working detector on this machine.

Important compatibility note:
- TensorRT `10.3` on JetPack 6 rejects the older Caffe-based sample models used
  by stock `imagenet` in `jetson-inference`
- result: the `jetson-inference` binaries build, but the default Caffe example
  model path fails at runtime

So the reliable path became:
- Argus camera input
- OpenCV DNN inference
- YOLO-family models loaded from either:
  - Darknet `weights + cfg`
  - ONNX

The first in-repo proof on this stack was `MobileNetV2`, but whole-frame
classification is the wrong primitive for posture work. YOLO person detection is
the right next layer.

----- Jetson Packages Installed -----
These packages were installed on the Jetson during bring-up:

- `cuda-nvcc-12-6`
- `python3-opencv`
- `python3-libnvinfer`
- `python3-libnvinfer-dev`
- `libnpp-dev-12-6`
- `cuda-libraries-dev-12-6`

Python environment fix applied on the Jetson:
- user-site `numpy` was downgraded from `2.2.6` to `1.26.4`

Why:
- `python3-opencv` on this Jetson was built against NumPy 1.x ABI
- with NumPy 2.x installed in `~/.local`, `cv2` import failed

Command used:
- `python3 -m pip install --user --upgrade 'numpy<2'`

----- How The Demo Works -----
Pipeline:
1. `nvarguscamerasrc` captures frames from the CSI camera.
2. GStreamer converts frames into BGR for OpenCV.
3. OpenCV loads a YOLO model.
4. Each frame is resized for the detector.
5. OpenCV DNN runs detection.
6. Bounding boxes are decoded back into camera-space coordinates.
7. Each detection is converted into a positional encoding vector.
8. Boxes plus normalized positions are drawn onto the frame.
9. Frames are written to an AVI file and detections are written to JSONL.

Camera pipeline string used by the script:
- `nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, framerate=(fraction)30/1, format=(string)NV12 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=true max-buffers=1 sync=false`

Default model files:
- weights: `/home/logan/models/yolov4-tiny.weights`
- config: `/home/logan/models/yolov4-tiny.cfg`

Optional ONNX path:
- model: `/home/logan/models/yolo*.onnx`

Positional encoding emitted per detection:
- normalized center: `cx`, `cy`
- normalized size: `w`, `h`
- normalized area
- sinusoidal position features: `sin(pi*cx)`, `cos(pi*cx)`, `sin(pi*cy)`, `cos(pi*cy)`
- confidence
- track-local motion features: `delta_cx`, `delta_cy`, `delta_area`, `speed`

This produces a compact vector for downstream logic:
- `[cx, cy, w, h, area, sinx, cosx, siny, cosy, score]`

Each JSONL line also carries:
- `primary_detection`: highest-confidence detection in that frame
- `track_id`: stable ID matched across nearby frames by IoU
- `motion.delta_center_xy_norm`
- `motion.delta_area_norm`
- `motion.speed_norm`
- `motion.age_frames`

----- Pose Estimation Demo -----
For posture work, the next visual step is the pose overlay demo:
- `jetson/inference/pose_camera_demo.py`

What it draws:
- body keypoints from a pretrained YOLO pose model
- arm and torso skeleton lines
- a green `chest` center dot from shoulder midpoint
- a magenta spine proxy line using `nose -> chest_center -> hip_center` when available
- live torso / lean metrics on the frame

What it records:
- `/home/logan/pose-demo.avi`
- `/home/logan/pose-detections.jsonl`

Default pose model path:
- `/home/logan/models/yolo11n-pose.onnx`

Model export path:
- `python3 -m pip install --user ultralytics`
- `yolo export model=yolo11n-pose.pt format=onnx opset=12 imgsz=640`
- `mv yolo11n-pose.onnx ~/models/yolo11n-pose.onnx`

Run:
- `python3 ~/jetson-runtime/jetson/inference/pose_camera_demo.py --frames 300 --output ~/pose-demo.avi`

Useful options:
- `--pose-out ~/pose-detections.jsonl`
- `--imgsz 640`
- `--kpt-thres 0.35`
- `--calibration-frames 45`

Important interpretation note:
- this does not directly observe the spine
- it infers a spine proxy from visible landmarks, mainly shoulders, hips, and nose
- because the camera angle is odd, the built-in calibration phase matters more than raw world-vertical angle

With the camera still pointed at the ceiling, detections may legitimately be
empty for now. That is expected. The useful validation at this stage is:
- the CSI stream stays healthy
- the YOLO detector runs
- `persons=0` is logged cleanly when nobody is in frame
- once the camera is aimed at a person, normalized position vectors appear

----- How To Run It On The Jetson -----
Copy the repo script to the Jetson or run it from a checkout there.

If the default model files are missing:
- `mkdir -p ~/models`
- `wget -O ~/models/yolov4-tiny.cfg https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg`
- `wget -O ~/models/yolov4-tiny.weights https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights`

Run:
- `python3 ~/jetson-runtime/jetson/inference/yolo_positional_camera_demo.py --frames 300 --output ~/yolo-positional-demo.avi`

Useful options:
- `--frames 300`
- `--width 1280`
- `--height 720`
- `--fps 30`
- `--imgsz 416`
- `--conf-thres 0.35`
- `--nms-thres 0.45`
- `--positions-out ~/yolo-positional-detections.jsonl`
- `--save-crops-dir ~/person-crops`
- `--save-crops-min-score 0.55`
- `--save-crops-every 10`
- `--track-match-iou 0.30`
- `--print-every 15`
- `--output ~/yolo-positional-demo.avi`
- `--all-classes`
- `--model ~/models/your-yolo.onnx`

Expected healthy output:
- terminal logs printing `persons=` every N frames
- a finished video file at the requested output path
- a JSONL file where each line contains decoded detections and position vectors
- optional saved person crops for fine-tuning

----- What To Review Tomorrow -----
If you want to understand this stack quickly, review these in order:

1. `jetson/camera/README.txt`
   Why: this explains how the CSI camera became available at all.

2. `jetson/inference/yolo_positional_camera_demo.py`
   Why: this is the shortest end-to-end example of capture -> preprocess ->
   decode -> positional encoding -> overlay -> save.

2b. `jetson/inference/pose_camera_demo.py`
   Why: this is the shortest end-to-end example of capture -> pose decode ->
   skeleton overlay -> chest/spine proxy -> save.

3. The GStreamer pipeline in `yolo_positional_camera_demo.py`
   Why: this is the Jetson-specific part. Understanding `nvarguscamerasrc`,
   `nvvidconv`, and `appsink` will help with every later camera task.

4. The YOLO decode path in `yolo_positional_camera_demo.py`
   Why: this is where model-specific geometry handling happens.

5. `jetson-inference` notes in this file
   Why: this explains why the stock sample path is not the primary path on this
   JetPack / TensorRT combination.

Questions worth answering as you review:
- Where does the CSI frame become a NumPy array?
- What assumptions does the model make about image size and normalization?
- Which part is Jetson-specific, and which part is generic CV / ML plumbing?
- What output format is most useful for collecting a posture dataset?
- Should the next step after person detection be pose estimation or cropped-person classification?

----- Recommended Next Step -----
The next technical step after this detector should be either:
- person detection + cropped posture classifier
- pose / keypoint estimation

Why:
- classification over the whole frame is a weak fit for posture
- posture work needs person-localized geometry first
- the positional vector gives a cheap, explicit representation of where the person is in frame

Practical next move:
- run the YOLO demo while the camera is still aimed at the ceiling to confirm the pipeline is stable
- point the camera at yourself and confirm `person` detections plus reasonable `cx/cy/w/h`
- if that works, enable `--save-crops-dir` and collect clean person crops for fine-tuning if needed
- if the base detector is already stable, the next step is pose estimation or a cropped-person classifier

If the pose overlay is stable on your camera angle, the best next step is:
- derive posture features from the pose JSONL
- calibrate against your upright seated baseline
- then train or tune a `good / okay / bad` scorer on top of those features
