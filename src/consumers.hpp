#pragma once
#include "llm.hpp"
#include "messages.hpp"
#include "time.hpp"

#include <iostream>
#include <string>

class ConsoleLcdConsumer {
  public:
    void consume(const DisplayText &msg) {
        std::cout << "[LCD]\n";
        std::cout << "  " << msg.line1 << "\n";
        std::cout << "  " << msg.line2 << "\n";
    }
};

class ConsoleSpeakerConsumer {
  public:
    void consume(const PlayTone &msg) {
        std::cout << "[SPEAKER] tone freq=" << msg.frequency_hz << "Hz dur=" << msg.duration_ms
                  << "ms\n";
    }
};

class ConsoleLogConsumer {
  public:
    void consume(const LogLine &msg) { std::cout << "[LOG] " << msg.text << "\n"; }
};

class LlmActionConsumer {
  public:
    explicit LlmActionConsumer(ILlmClient &llm) : llm_(llm) {}

    LlmResponse consume(const LlmPrompt &msg) {
        LlmRequest req;
        req.seq = seq_++;
        req.t_request_ns = now_ns();
        req.prompt = msg.prompt;
        return llm_.infer(req);
    }

  private:
    ILlmClient &llm_;
    uint64_t seq_{0};
};
