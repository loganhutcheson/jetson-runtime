#pragma once
#include <any>
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// Dummy: base class for subscriber queues (type-erased)
struct ISubQueue {
    virtual ~ISubQueue() = default;
    virtual void push_any(std::any msg) = 0;
};

// Dummy: typed subscriber queue wrapper
template <typename T> struct SubQueue : ISubQueue {
    // Use your RingBuffer<T> internally
    // RingBuffer<T> q;
    // For skeleton purposes:
    std::mutex mu;
    std::vector<T> buffer; // placeholder

    void push_any(std::any msg) override {
        // convert any->T and enqueue
        if (msg.type() == typeid(T)) {
            std::lock_guard<std::mutex> lk(mu);
            buffer.push_back(std::any_cast<T>(std::move(msg)));
            // if bounded: drop oldest / count drops
        }
    }

    // Pull everything since last tick
    std::vector<T> drain() {
        std::lock_guard<std::mutex> lk(mu);
        std::vector<T> out;
        out.swap(buffer);
        return out;
    }
};

// In-process bus: topic -> list of subscriber queues
class Bus {
  public:
    template <typename T> std::shared_ptr<SubQueue<T>> subscribe(const std::string &topic) {
        auto q = std::make_shared<SubQueue<T>>();
        std::lock_guard<std::mutex> lk(mu_);
        topics_[topic].push_back(q);
        return q;
    }

    template <typename T> void publish(const std::string &topic, T msg) {
        std::vector<std::shared_ptr<ISubQueue>> subs;
        {
            std::lock_guard<std::mutex> lk(mu_);
            auto it = topics_.find(topic);
            if (it == topics_.end())
                return;
            subs = it->second; // copy list to publish without holding lock
        }
        std::any a = std::move(msg);
        for (auto &sub : subs) {
            sub->push_any(a); // fanout
        }
    }

  private:
    std::mutex mu_;
    std::unordered_map<std::string, std::vector<std::shared_ptr<ISubQueue>>> topics_;
};
