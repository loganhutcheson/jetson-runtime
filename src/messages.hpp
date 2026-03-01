#pragma once
#include <cstdint>
#include <string>
#include <vector>

inline uint64_t now_ns();

struct CameraFrame {
    uint64_t seq = 0;
    uint64_t t_capture_ns = 0;
    int width = 640;
    int height = 480;
    // dummy payload (optional)
    std::vector<uint8_t> bytes;
};

struct ImuSample {
    uint64_t seq = 0;
    uint64_t t_capture_ns = 0;
    float ax = 0, ay = 0, az = 9.81f;
    float gx = 0, gy = 0, gz = 0;
};

struct LlmRequest {
    uint64_t seq = 0;
    uint64_t t_request_ns = 0;
    std::string prompt;
};

struct LlmResponse {
    uint64_t seq = 0;
    uint64_t t_response_ns = 0;
    std::string text;
    // later: tokens/sec, latency, etc.
};
