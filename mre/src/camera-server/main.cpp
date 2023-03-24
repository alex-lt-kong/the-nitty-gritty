#include <signal.h>
#include <thread>
#include <vector>

#include <spdlog/spdlog.h>

#include "deviceManager.h"

using namespace std;


static vector<deviceManager> myDevices;
/*
void signal_handler(int signum) {
    if (signum == SIGPIPE) {    
        return;
    }
    //spdlog::warn("Signal: {} caught, all threads will quit gracefully", signum);
    for (size_t i = 0; i < myDevices.size(); ++i) {
        myDevices[i].StopInternalEventLoopThread();
        //spdlog::info("{}-th device: StopInternalEventLoopThread() called", i);
    }
}

void register_signal_handlers() {
    struct sigaction act;
    act.sa_handler = signal_handler;
    sigemptyset(&act.sa_mask);
    act.sa_flags = SA_RESETHAND;
    if (sigaction(SIGINT, &act, 0) + sigaction(SIGABRT, &act, 0) +
        sigaction(SIGTERM, &act, 0) + sigaction(SIGPIPE, &act, 0) < 0) {
        throw runtime_error("sigaction() called failed, errno: " +
            to_string(errno));
    }
}*/

int main() {
    spdlog::set_pattern("%Y-%m-%dT%T.%e%z|%5t|%8l| %v");
    spdlog::info("cs started"); 
    //register_signal_handlers();


    size_t deviceCount =3;

    myDevices = vector<deviceManager>();
    myDevices.reserve(deviceCount);
    for (size_t i = 0; i < deviceCount; ++i) {
        myDevices.emplace_back(i);
        myDevices.back().StartInternalEventLoopThread();
    }
    spdlog::info("{} threads started", deviceCount);
    for (size_t i = 0; i < myDevices.size(); ++i) {
        myDevices[i].WaitForInternalEventLoopThreadToExit();
    }
    return 0;  
}
