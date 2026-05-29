//
// Created by robin on 10.09.24.
//

#ifndef XCUT_TIMER_H
#define XCUT_TIMER_H

#include <chrono>

class Timer {
  std::chrono::time_point<std::chrono::steady_clock, std::chrono::duration<double>> startTime;
public:
    inline void start();
    inline double timeSeconds();
    inline long long timeMillis();
    inline long long timeMicros();
    inline long long timeNanos();
};

void Timer::start() {
    startTime = std::chrono::steady_clock::now();
}

double Timer::timeSeconds() {
    auto end = std::chrono::steady_clock::now(); // End time
    std::chrono::duration<double> elapsed_seconds = end - startTime; // Calculate elapsed time
    return elapsed_seconds.count(); // Return sorting time in seconds
}

long long Timer::timeMillis() {
    auto end = std::chrono::steady_clock::now(); // End time
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - startTime);
    return ms.count(); // Return sorting time in seconds
}

long long Timer::timeMicros() {
    auto end = std::chrono::steady_clock::now(); // End time
    auto ms = std::chrono::duration_cast<std::chrono::microseconds>(end - startTime);
    return ms.count(); // Return sorting time in seconds
}

long long Timer::timeNanos() {
    auto end = std::chrono::steady_clock::now(); // End time
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - startTime);
    return ns.count(); // Return sorting time in seconds
}

#endif // XCUT_TIMER_H
