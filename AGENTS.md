# Jetson Runtime Notes

## Jetson Orin Nano IMU bring-up
- On the Jetson Orin Nano dev kit 40-pin header, physical pins `3/5` are the active I2C pair for the MPU-6050 bring-up, but on Linux they show up on `/dev/i2c-7`.
- Probing `/dev/i2c-1` can be misleading on this platform because the header I2C bus for pins `3/5` is exposed as bus `7`.
- `i2cdetect -y -r 7` is the quickest validation step for header pins `3/5`.
- With `AD0` tied to ground, the expected MPU-6050 address is `0x68`.

## Runtime defaults
- The repo defaults the MPU runtime config to `/dev/i2c-7` and `0x68`.
- The real IMU path is enabled with:
  - `JETSON_IMU_SOURCE=mpu6050`
  - `JETSON_IMU_ENABLE=1`
  - `JETSON_IMU_I2C_DEV=/dev/i2c-7`
  - `JETSON_IMU_I2C_ADDR=0x68`

## Jetson validation
- If `cmake` is missing on the Jetson, the project can still be compiled directly with:
  - `/usr/bin/c++ -std=c++20 -O2 -Isrc src/main.cpp -pthread -o llm_experiments`
- A healthy real-IMU startup looks like:
  - `[main] IMU source=mpu6050(/dev/i2c-7, addr=0x68)`
- A healthy periodic IMU log looks like:
  - `[imu] hz=... total=... dropped=0 sample.seq=... accel=(ax,ay,az) gyro=(gx,gy,gz)`
