#include "bus.hpp"
#include "planner.hpp"
#include "producers.hpp"

int main() {
    Bus bus;

    CameraProducer cam(bus, 30);
    ImuProducer imu(bus, 200);
    PlannerNode planner(bus);

    cam.start();
    imu.start();

    for (int i = 0; i < 60; i++) { // 30s at 2Hz
        planner.tick_2hz();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    imu.stop();
    cam.stop();
}
