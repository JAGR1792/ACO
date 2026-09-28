#pragma once
#include <sys/resource.h>
#include <chrono>

inline long peak_rss_kb() {
  struct rusage ru;
  getrusage(RUSAGE_SELF, &ru);
  return ru.ru_maxrss;  // Linux: KB
}

struct Timer {
  std::chrono::steady_clock::time_point t0;
  Timer() { reset(); }
  void reset() { t0 = std::chrono::steady_clock::now(); }
  double ms() const {
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
  }
};
