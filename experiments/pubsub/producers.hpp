#pragma once
#include "bus.hpp"
#include "messages.hpp"
#include "time.hpp"
#include <atomic>
#include <chrono>
#include <thread>

class ImuProducer {
  public:
    ImuProducer(Bus &bus, int hz = 200) : bus_(bus), period_(1'000'000 / hz) {}

    void start() {
        running_.store(true);
        th_ = std::thread([this] { run(); });
    }
    void stop() {
        running_.store(false);
        if (th_.joinable())
            th_.join();
    }

  private:
    void run() {
        while (running_.load()) {
            ImuSample s;
            s.seq = seq_++;
            s.t_ns = now_ns();
            s.az = 9.81f;
            bus_.publish("/imu", std::move(s));
            std::this_thread::sleep_for(std::chrono::microseconds(period_));
        }
    }

    Bus &bus_;
    int period_;
    std::atomic<bool> running_{false};
    std::thread th_;
    uint64_t seq_{0};
};

class CameraProducer {
  public:
    CameraProducer(Bus &bus, int fps = 30) : bus_(bus), period_(1'000'000 / fps) {}

    void start() {
        running_.store(true);
        th_ = std::thread([this] { run(); });
    }
    void stop() {
        running_.store(false);
        if (th_.joinable())
            th_.join();
    }

  private:
    void run() {
        while (running_.load()) {
            CameraFrame f;
            f.seq = seq_++;
            f.t_ns = now_ns();
            f.bytes.resize(640 * 480, 0);
            bus_.publish("/camera", std::move(f));
            std::this_thread::sleep_for(std::chrono::microseconds(period_));
        }
    }

    Bus &bus_;
    int period_;
    std::atomic<bool> running_{false};
    std::thread th_;
    uint64_t seq_{0};
};
