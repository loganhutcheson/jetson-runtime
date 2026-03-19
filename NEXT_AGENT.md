# Next Agent Handoff

Date:
- 2026-03-16
- 2026-03-17

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

Jetson validation completed on 2026-03-17:
- SSH login shell still reproduced the Torch import failure at first.
- installed `libnvtoolsext1` on the Jetson:
  - `sudo apt-get install -y libnvtoolsext1`
- after that, `torch` advanced further but still failed on:
  - `ImportError: libcudnn.so.8: cannot open shared object file`
- the Jetson currently has cuDNN 9 runtime libraries, not cuDNN 8:
  - `/usr/lib/aarch64-linux-gnu/libcudnn.so.9`
- this means the current `torch 2.1.0` install under `/usr/local/lib/python3.10/dist-packages` is not matched to the JetPack 6 cuDNN stack
- despite the Torch mismatch, the OpenCV + Darknet positional workflow does run end-to-end on the Jetson:
  - `python3 ~/jetson-runtime/jetson/inference/yolo_positional_camera_demo.py --frames 60 --print-every 10 --output ~/yolo-positional-demo.avi --positions-out ~/yolo-positional-detections.jsonl`
- observed result:
  - CSI camera opened successfully on `sensor-id=0`
  - output files were written:
    - `/home/logan/yolo-positional-demo.avi`
    - `/home/logan/yolo-positional-detections.jsonl`
  - run completed with `done frames=60 ... fps=4.10`
  - detections were empty during this SSH run (`persons=0`), consistent with the camera not currently being aimed at a person

Jetson pose runtime progress completed on 2026-03-18:
- installed a JetPack-6-compatible user-site Torch stack from the Jetson AI Lab JP6 / CUDA 12.6 index:
  - `torch 2.10.0`
  - `torchvision 0.25.0`
  - `nvidia-cudss-cu12 0.7.1.6`
- also installed:
  - `cuda-cupti-12-6` via `apt`
  - `sympy 1.14.0` via `pip`
- added a global linker config on the Jetson so `python3 -c 'import torch'` works without a manual `LD_LIBRARY_PATH` export:
  - `/etc/ld.so.conf.d/jetson-logan-nvidia-cu12.conf`
- verified on-device:
  - `python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)'`
  - output was `2.10.0 True 12.6`
- exported both pose checkpoints on-device:
  - `/home/logan/models/yolo11n-pose.pt`
  - `/home/logan/models/yolov8n-pose.pt`
  - `/home/logan/models/yolo11n-pose.onnx`
- important finding:
  - OpenCV 4.5.4 on this Jetson still fails at `net.forward()` for the exported pose ONNX graphs
  - exact error:
    - `cv2.error: ... shape_utils.hpp:170: error: (-215:Assertion failed) ... in function 'total'`
- working fix:
  - `jetson/inference/pose_camera_demo.py` now supports an Ultralytics `.pt` backend for live Jetson inference
- verified working command on the Jetson:
  - `python3 ~/jetson-runtime/jetson/inference/pose_camera_demo.py --backend ultralytics --model ~/models/yolov8n-pose.pt --frames 60 --print-every 10 --output ~/pose-demo-ultra.avi --pose-out ~/pose-detections-ultra.jsonl`
- observed result:
  - CSI camera opened successfully on `sensor-id=0`
  - run completed with `done frames=60 elapsed=8.71s fps=6.89`
  - detections were empty during this SSH run (`have_pose=0`), again consistent with camera aim during the remote test

What the next agent should do first:
1. Keep using `yolo_positional_camera_demo.py` as the working on-device path for live positional encoding capture.
2. Point the CSI camera at a person and rerun the YOLO positional script to verify non-empty person detections and position vectors.
3. For live pose on this Jetson, prefer the new Ultralytics backend instead of the OpenCV ONNX path:
   - `python3 ~/jetson-runtime/jetson/inference/pose_camera_demo.py --backend ultralytics --model ~/models/yolov8n-pose.pt --frames 300 --output ~/pose-demo-ultra.avi --pose-out ~/pose-detections-ultra.jsonl`
4. Only keep pursuing the ONNX path if you also plan to upgrade / rebuild the Jetson OpenCV DNN stack.
5. Review the pose overlay quality for:
   - shoulders
   - elbows / wrists
   - hips
   - chest center
   - spine proxy stability

Notes:
- there is an unrelated untracked `temp/` directory in the local repo; it was intentionally left alone
- do not assume the top-level `README.txt` on the Jetson checkout is up to date unless it is recopied from this commit
