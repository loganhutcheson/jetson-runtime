#pragma once
#include <mutex>
#include <optional>

template <typename T> class Latest {
  public:
    void publish(T v) {
        std::lock_guard<std::mutex> lk(mu_);
        value_ = std::move(v);
    }

    // take latest and clear it (so consumer knows if it got something new)
    std::optional<T> take() {
        std::lock_guard<std::mutex> lk(mu_);
        if (!value_)
            return std::nullopt;
        auto out = std::move(value_);
        value_.reset();
        return out;
    }

    // peek without clearing
    std::optional<T> peek() const {
        std::lock_guard<std::mutex> lk(mu_);
        return value_;
    }

  private:
    mutable std::mutex mu_;
    std::optional<T> value_;
};
