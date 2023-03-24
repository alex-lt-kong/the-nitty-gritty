#include <iostream>
#include <spdlog/spdlog.h>

#include "deviceManager.h"

deviceManager::deviceManager(const size_t deviceIndex) {

    //spdlog::set_pattern(logFormat);
    this->tid = deviceIndex;
}


void deviceManager::InternalThreadEntry() {
    
    while (!_internalThreadShouldQuit) {
        spdlog::info("from thread: {}", tid);
        usleep(500000);
    }
    spdlog::info("[{}] thread quits gracefully", tid);
}
