# Next Agent Handoff

Date:
- 2026-03-16

What was added locally:
- `jetson/inference/yolo_positional_camera_demo.py`
- `jetson/inference/pose_camera_demo.py`
- updated `jetson/inference/README.txt`
- updated `README.txt`

What the new scripts do:
- `yolo_positional_camera_demo.py`
  - YOLO-based person detection on the Jetson CSI camera feed
  - emits normalized positional encodings
  - adds lightweight `track_id` and frame-to-frame motion deltas
  - can optionally save person crops for future fine-tuning
- `pose_camera_demo.py`
  - YOLO pose ONNX demo on the same CSI pipeline
  - draws skeleton lines, chest center, and a spine proxy
  - writes pose JSONL plus annotated output video
  - includes a short calibration phase for odd camera angle baselining

Local verification completed:
- `python3 -m py_compile jetson/inference/yolo_positional_camera_demo.py`
- `python3 -m py_compile jetson/inference/pose_camera_demo.py`

Jetson access used today:
- `ssh logan@jetson.local`
- password supplied by user during this session: `Lablab123`

Jetson state observed today:
- hostname: `jetson`
- Python: `3.10.12`
- OpenCV: `4.5.4`
- NumPy: `1.26.4`
- repo already exists at `~/jetson-runtime`
- `~/models` already contained:
  - `mobilenetv2-7.onnx`
  - `yolov4-tiny.cfg`
  - `yolov4-tiny.weights`

Jetson actions already completed:
- copied these files to the Jetson repo checkout:
  - `~/jetson-runtime/jetson/inference/pose_camera_demo.py`
  - `~/jetson-runtime/jetson/inference/yolo_positional_camera_demo.py`
- remote syntax check succeeded:
  - `python3 -m py_compile ~/jetson-runtime/jetson/inference/pose_camera_demo.py ~/jetson-runtime/jetson/inference/yolo_positional_camera_demo.py`
- installed `ultralytics` on the Jetson with:
  - `python3 -m pip install --user --no-deps ultralytics`

Important remote blocker discovered:
- importing `torch` over non-interactive SSH currently fails on the Jetson with missing CUDA library resolution:
  - `libnvToolsExt.so.1`
  - `libcublas.so.*[0-9]`
- this means `ultralytics` import/export did not complete during this session
- likely cause: the SSH shell environment is missing the CUDA library path, or the installed Torch build expects paths not set in non-login sessions

What the next agent should do first:
1. SSH to the Jetson and inspect CUDA lib locations plus `LD_LIBRARY_PATH` in an interactive login shell.
2. Verify whether `python3 -c 'import torch'` works directly on-device at the Jetson console.
3. If Torch is healthy interactively, export `yolo11n-pose.onnx` on the Jetson or download a compatible pre-exported pose ONNX.
4. Run:
   - `python3 ~/jetson-runtime/jetson/inference/pose_camera_demo.py --frames 300 --output ~/pose-demo.avi --pose-out ~/pose-detections.jsonl`
5. Review the pose overlay quality for:
   - shoulders
   - elbows / wrists
   - hips
   - chest center
   - spine proxy stability

Notes:
- there is an unrelated untracked `temp/` directory in the local repo; it was intentionally left alone
- do not assume the top-level `README.txt` on the Jetson checkout is up to date unless it is recopied from this commit
