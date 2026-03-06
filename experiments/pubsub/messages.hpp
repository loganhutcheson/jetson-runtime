#pragma once
#include <cstdint>
#include <string>
#include <vector>

struct ImuSample {
    uint64_t seq{};
    uint64_t t_ns{};
    float ax{}, ay{}, az{};
};

struct CameraFrame {
    uint64_t seq{};
    uint64_t t_ns{};
    std::vector<uint8_t> bytes; // dummy
};

struct LlmRequest {
    uint64_t seq{};
    uint64_t t_ns{};
    std::string prompt;
};

struct LlmResponse {
    uint64_t seq{};
    uint64_t t_ns{};
    std::string text;
};
