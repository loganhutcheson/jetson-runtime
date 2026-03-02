#include "latest.hpp"
#include "llm.hpp"
#include "messages.hpp"
#include "producers.hpp"
#include "time.hpp"

#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>

int main() {
    Latest<CameraFrame> cam_latest;
    RingBuffer<ImuSample> imu_buf;

    CameraProducer cam(cam_latest, 30);
    ImuProducer imu(imu_buf, 200);

    StubLlmClient llm;

    cam.start();
    imu.start();

    std::ofstream csv("results_mac.csv");
    csv << "t_ns,cam_seq,cam_age_ms,imu_seq,imu_age_ms,imu_cnt,llm_seq,llm_latency_ms\n";

    uint64_t llm_seq = 0;

    // Main loop: 2Hz “planner” tick (LLM is slow)
    for (int i = 0; i < 30; i++) {
        uint64_t t0 = now_ns();
        static uint64_t last_tick_ns = 0;
        double loop_ms = -1;

        if (last_tick_ns != 0) {
            loop_ms = (double)(t0 - last_tick_ns) / 1e6;
        }
        last_tick_ns = t0;

        auto cam_opt = cam_latest.peek(); // latest frame (don’t clear)
        std::vector<ImuSample> imu_samples;
        auto imu_cnt = imu_buf.popall(imu_samples); // latest imu

        uint64_t cam_seq = cam_opt ? cam_opt->seq : 0;
        uint64_t imu_seq = imu_cnt > 0 ? imu_samples.back().seq : 0;

        double cam_age_ms = cam_opt ? (double)(t0 - cam_opt->t_capture_ns) / 1e6 : -1;
        double imu_age_ms = imu_cnt > 0 ? (double)(t0 - imu_samples.back().t_capture_ns) / 1e6 : -1;

        LlmRequest req;
        req.seq = llm_seq++;
        req.t_request_ns = t0;
        req.prompt = "cam_seq=" + std::to_string(cam_seq) + " imu_seq=" + std::to_string(imu_seq);

        auto resp = llm.infer(req);
        double llm_latency_ms = (double)(resp.t_response_ns - req.t_request_ns) / 1e6;

        csv << t0 << "," << cam_seq << "," << cam_age_ms << "," << imu_seq << "," << imu_age_ms
            << "," << imu_cnt << "," << resp.seq << "," << llm_latency_ms << "\n";
        std::cout << "tick " << i << " imu_cnt=" << imu_cnt << " cam_age_ms=" << cam_age_ms
                  << " imu_age_ms=" << imu_age_ms << " llm_latency_ms=" << llm_latency_ms
                  << " loop_ms=" << loop_ms << "\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    cam.stop();
    imu.stop();

    std::cout << "wrote results_mac.csv\n";
    return 0;
}
