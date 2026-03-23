----- Jetson Camera Inference Notes -----
This directory now documents the YOLO-based camera inference path we want to
carry forward on the Jetson Orin Nano after the CAM1 IMX219 bring-up.

Date updated:
- 2026-03-16
- 2026-03-17

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

Validated again on 2026-03-17:
- the positional demo ran successfully over SSH on the Jetson with:
  - `python3 ~/jetson-runtime/jetson/inference/yolo_positional_camera_demo.py --frames 60 --print-every 10 --output ~/yolo-positional-demo.avi --positions-out ~/yolo-positional-detections.jsonl`
- Argus opened the active camera on `sensor-id=0`
- the run completed and wrote both output files
- detections were empty during that validation run because the camera was not pointed at a person at the time

Working repo script:
- `jetson/inference/yolo_positional_camera_demo.py`
- `jetson/inference/pose_camera_demo.py`

Pose runtime status on 2026-03-18:
- the Jetson now has a working JP6-compatible Torch stack in user site
- `python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)'`
  returned:
  - `2.10.0 True 12.6`
- OpenCV DNN still failed on both exported pose ONNX graphs on this Jetson's `cv2 4.5.4`
- the working live pose path is now the same `pose_camera_demo.py` script using its Ultralytics backend with a `.pt` model

----- 30-Minute Recording Plan -----
If your goal is to train `bad / okay / good` posture from this exact camera angle,
collect one clean 30-minute pose dataset before worrying about model training.

What to record:
- 10 minutes of `good`
- 10 minutes of `okay`
- 10 minutes of `bad`

Why this split:
- it gives balanced class time
- it keeps labeling simple
- it matches the current camera angle and desk setup instead of mixing viewpoints

Before you start:
1. Keep the camera physically fixed.
2. Keep the chair, desk, and monitor in their normal positions.
3. Turn the lights on and leave them stable.
4. Sit where you normally work.
5. Avoid changing the zoom, crop, or camera mount between class recordings.

Recommended directory layout on the Jetson:
- `mkdir -p ~/posture-data/good ~/posture-data/okay ~/posture-data/bad`

Recommended live pose command template:
- `python3 ~/jetson-runtime/jetson/inference/pose_camera_demo.py --backend ultralytics --model ~/models/yolov8n-pose.pt --frames 54000 --output ~/posture-data/<label>/<label>.avi --pose-out ~/posture-data/<label>/<label>.jsonl`

Why `54000` frames:
- `30 minutes * 60 seconds * 30 fps = 54000`

Practical recommendation:
- do not actually record all 30 minutes in one file
- record three separate 10-minute sessions instead
- that makes labels unambiguous and makes failed runs cheaper to redo

10-minute per-class commands:
- `python3 ~/jetson-runtime/jetson/inference/pose_camera_demo.py --backend ultralytics --model ~/models/yolov8n-pose.pt --frames 18000 --output ~/posture-data/good/good_001.avi --pose-out ~/posture-data/good/good_001.jsonl`
- `python3 ~/jetson-runtime/jetson/inference/pose_camera_demo.py --backend ultralytics --model ~/models/yolov8n-pose.pt --frames 18000 --output ~/posture-data/okay/okay_001.avi --pose-out ~/posture-data/okay/okay_001.jsonl`
- `python3 ~/jetson-runtime/jetson/inference/pose_camera_demo.py --backend ultralytics --model ~/models/yolov8n-pose.pt --frames 18000 --output ~/posture-data/bad/bad_001.avi --pose-out ~/posture-data/bad/bad_001.jsonl`

How to perform each class:
- `good`:
  - sit tall in the position you actually want to maintain
  - shoulders level
  - chest open
  - head roughly centered above torso
- `okay`:
  - your normal, acceptable working posture
  - not ideal, but not strongly slouched
  - slight lean or asymmetry is fine
- `bad`:
  - the posture you want the system to catch
  - for example forward head, rounded shoulders, or chest collapse
  - use realistic bad posture, not a cartoon exaggeration

What to do during recording:
- type, read, look at the screen, and make small natural movements
- include slight shifts and fidgets
- do not freeze in one pose for the full recording
- stay inside the camera view

What not to do:
- do not mix multiple labels in one recording
- do not move the camera between recordings
- do not change lighting halfway through
- do not let another person enter frame if you can avoid it

Quick validation after each recording:
1. Confirm the `.avi` file exists.
2. Confirm the `.jsonl` file exists.
3. Check the first few lines:
   - `sed -n '1,3p' ~/posture-data/good/good_001.jsonl`
4. Check the last line:
   - `tail -n 1 ~/posture-data/good/good_001.jsonl`
5. Make sure `primary_detection` is usually not `null`.

Minimum acceptable dataset for the next step:
- at least one 10-minute file for each of:
  - `good`
  - `okay`
  - `bad`

Better dataset:
- 2 to 3 recordings per class on different days
- same camera angle
- same person
- slightly different shirts / lighting / time of day

What we will train from later:
- primarily the pose JSONL, not raw pixels first
- features like:
  - chest center
  - hip center
  - torso angle
  - shoulder tilt
  - nose-to-chest relation
  - short time-window motion and stability

What to read tomorrow before prompting again:
1. This `30-Minute Recording Plan` section.
2. `----- Pose Estimation Demo -----` below.
3. `jetson/inference/pose_camera_demo.py`

When you come back next, the useful handoff is:
- which recordings you captured
- where they are stored
- whether any recordings had `primary_detection: null` too often
- whether your `good / okay / bad` definitions need to be tightened

Default model path:
- `yolov4-tiny` via Darknet weights + cfg because it is easy to fetch on this Jetson

Optional model path:
- YOLO-family ONNX is also supported by the same script

Pose overlay path:
- a separate YOLO pose ONNX demo is available for real-time landmarks and spine proxy visualization
- the same script now also supports a Jetson-tested Ultralytics `.pt` backend for live inference when OpenCV DNN is not compatible with the exported ONNX

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

Current Jetson blocker for this export path:
- the installed Torch on the Jetson is currently mismatched to the JetPack 6 runtime stack
- after installing `libnvtoolsext1`, `import torch` still fails with:
  - `ImportError: libcudnn.so.8: cannot open shared object file`
- the machine currently exposes cuDNN 9:
  - `/usr/lib/aarch64-linux-gnu/libcudnn.so.9`
- practical implication:
  - `ultralytics`-based ONNX export is still blocked on-device until Torch is replaced with a JetPack-compatible build
  - the fastest workaround is to export `yolo11n-pose.onnx` off-device and copy it into `~/models`

Update on 2026-03-18:
- this blocker is now resolved for Torch itself on the Jetson
- installed pieces:
  - `torch 2.10.0`
  - `torchvision 0.25.0`
  - `nvidia-cudss-cu12 0.7.1.6`
  - `cuda-cupti-12-6`
  - `sympy 1.14.0`
- a linker config was added on the Jetson so `libcudss.so.0` resolves without a manual shell export:
  - `/etc/ld.so.conf.d/jetson-logan-nvidia-cu12.conf`
- ONNX export now succeeds on-device for:
  - `/home/logan/models/yolo11n-pose.onnx`
- however, OpenCV DNN on this Jetson still crashes at `net.forward()` for the exported pose models, so the runtime recommendation changed:
  - use the Ultralytics `.pt` backend for live pose
  - treat the ONNX export as useful for later experimentation, not the primary working path on this Jetson today

Run:
- `python3 ~/jetson-runtime/jetson/inference/pose_camera_demo.py --frames 300 --output ~/pose-demo.avi`

Jetson-tested live pose command:
- `python3 ~/jetson-runtime/jetson/inference/pose_camera_demo.py --backend ultralytics --model ~/models/yolov8n-pose.pt --frames 300 --output ~/pose-demo-ultra.avi --pose-out ~/pose-detections-ultra.jsonl`

Useful options:
- `--pose-out ~/pose-detections.jsonl`
- `--imgsz 640`
- `--kpt-thres 0.35`
- `--calibration-frames 45`
- `--backend ultralytics`
- `--model ~/models/yolov8n-pose.pt`
- `--device 0`

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
