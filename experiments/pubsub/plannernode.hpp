#pragma once
#include "bus.hpp"
#include "messages.hpp"
#include "time.hpp"
#include <chrono>
#include <thread>

class PlannerNode {
  public:
    PlannerNode(Bus &bus)
        : bus_(bus), imu_sub_(bus_.subscribe<ImuSample>("/imu")),
          cam_sub_(bus_.subscribe<CameraFrame>("/camera")) {}

    void tick_2hz() {
        // Drain what arrived since last tick
        auto imu = imu_sub_->drain(); // vector<ImuSample>
        auto cam = cam_sub_->drain(); // vector<CameraFrame>

        // “Latest” policy can be applied here:
        CameraFrame *latest_cam = cam.empty() ? nullptr : &cam.back();

        // “Batch” policy for IMU:
        // fuse imu samples, compute stats, integrate, etc.
        size_t imu_n = imu.size();

        LlmRequest req;
        req.seq = seq_++;
        req.t_ns = now_ns();
        req.prompt = "cam=" + std::to_string(latest_cam ? latest_cam->seq : 0) +
                     " imu_count=" + std::to_string(imu_n);

        bus_.publish("/llm/request", std::move(req));
    }

  private:
    Bus &bus_;
    std::shared_ptr<SubQueue<ImuSample>> imu_sub_;
    std::shared_ptr<SubQueue<CameraFrame>> cam_sub_;
    uint64_t seq_{0};
};
