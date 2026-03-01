#pragma once
#include <mutex>
#include <optional>
#include <queue>

template <typename T> class RingBuffer {
  public:
    void push(T v) {
        std::lock_guard<std::mutex> lk(mu_);
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

  private:
    mutable std::mutex mu_;
    std::queue<T> q_;
};
