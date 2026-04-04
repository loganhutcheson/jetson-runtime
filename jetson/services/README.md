# Posture Runtime Service

Install the service on the Jetson:

```bash
cd /home/logan/jetson-runtime
./jetson/services/install_posture_runtime_service.sh
sudo systemctl enable --now posture-runtime.service
```

Edit the runtime configuration in:

```bash
/etc/default/posture-runtime
```

Recommended always-on defaults:

- `POSTURE_RUNTIME_FRAMES=0` to run forever
- empty `POSTURE_RUNTIME_OUTPUT` to disable AVI recording
- empty `POSTURE_RUNTIME_POSE_OUT` to disable JSONL recording
- set `POSTURE_RUNTIME_SQLITE_DB=/home/logan/posture/posture_history.sqlite3` for durable history
- keep `POSTURE_RUNTIME_POSTURE_MODEL=/tmp/posture_classifier.json` or replace it with a persistent model path

Developer flow:

```bash
sudo systemctl stop posture-runtime.service
cd /home/logan/jetson-runtime
./jetson/services/run_posture_runtime.sh
```

Return to service mode:

```bash
sudo systemctl start posture-runtime.service
sudo systemctl status posture-runtime.service
```

## Dashboard Service

Install the dashboard service on the Jetson:

```bash
cd /home/logan/jetson-runtime
./jetson/services/install_posture_dashboard_service.sh
sudo systemctl enable --now posture-dashboard.service
```

Edit the dashboard configuration in:

```bash
/etc/default/posture-dashboard
```

By default it serves the SQLite history on:

```bash
http://<jetson-host>:8787/
```
