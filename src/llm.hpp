#pragma once
#include "messages.hpp"
#include <string>

class ILlmClient {
  public:
    virtual ~ILlmClient() = default;
    virtual LlmResponse infer(const LlmRequest &req) = 0;
};

// Stub: deterministic “fake model”
class StubLlmClient : public ILlmClient {
  public:
    LlmResponse infer(const LlmRequest &req) override {
        LlmResponse r;
        r.seq = req.seq;
        r.t_response_ns = now_ns();
        r.text = "stubbed_response: " + req.prompt;
        return r;
    }
};
