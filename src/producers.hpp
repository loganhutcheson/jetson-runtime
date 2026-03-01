#pragma once
#include "latest.hpp"
#include "messages.hpp"
#include "ringbuffer.hpp"
#include "time.hpp"

#include <atomic>
#include <chrono>
#include <thread>

class CameraProducer {
  public:
    CameraProducer(Latest<CameraFrame> &out, int fps = 30)
        : out_(out), period_(std::chrono::microseconds(1'000'000 / fps)) {}

    void start() {
        running_ = true;
        th_ = std::thread([this] { run(); });
    }
    void stop() {
        running_ = false;
        if (th_.joinable())
            th_.join();
    }

  private:
    void run() {
        while (running_) {
            CameraFrame f;
            f.seq = seq_++;
            f.t_capture_ns = now_ns();
            // optional dummy payload
            f.bytes.resize(640 * 480, 0);
            out_.publish(std::move(f));
            std::this_thread::sleep_for(period_);
        }
    }

    Latest<CameraFrame> &out_;
    std::chrono::microseconds period_;
    std::atomic<bool> running_{false};
    std::thread th_;
    uint64_t seq_{0};
};

class ImuProducer {
  public:
    ImuProducer(RingBuffer<ImuSample> &out, int hz = 200)
        : out_(out), period_(std::chrono::microseconds(1'000'000 / hz)) {}

    void start() {
        running_ = true;
        th_ = std::thread([this] { run(); });
    }
    void stop() {
        running_ = false;
        if (th_.joinable())
            th_.join();
    }

  private:
    void run() {
        while (running_) {
            ImuSample s;
            s.seq = seq_++;
            s.t_capture_ns = now_ns();
            // dummy values; later you’ll read real IMU
            s.az = 9.81f;
            out_.push(std::move(s));
            std::this_thread::sleep_for(period_);
        }
    }

    RingBuffer<ImuSample> &out_;
    std::chrono::microseconds period_;
    std::atomic<bool> running_{false};
    std::thread th_;
    uint64_t seq_{0};
};
