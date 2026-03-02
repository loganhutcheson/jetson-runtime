#pragma once
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <queue>
#include <vector>

template <typename T> class RingBuffer {
  public:
    explicit RingBuffer(size_t cap = 256) : cap_(cap) {}

    void push(T v) {
        std::lock_guard<std::mutex> lk(mu_);
        if (q_.size() >= cap_) {
            q_.pop();
            dropped_++;
        }
        q_.push(std::move(v));
    }

    std::optional<T> peek() const {
        std::lock_guard<std::mutex> lk(mu_);
        if (q_.empty())
            return std::nullopt;
        return q_.front();
    }

    void pop() {
        std::lock_guard<std::mutex> lk(mu_);
        if (!q_.empty())
            q_.pop();
    }

    size_t popall(std::vector<T> &out) {
        std::lock_guard<std::mutex> lk(mu_);
        size_t count = 0;
        while (!q_.empty()) {
            out.push_back(std::move(q_.front()));
            q_.pop();
            count++;
        }

        return count;
    }

  private:
    mutable std::mutex mu_;
    std::queue<T> q_;
    size_t cap_;
    uint64_t dropped_;
};
