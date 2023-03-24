#ifndef DEVICE_MANAGER_H
#define DEVICE_MANAGER_H

#include <linux/stat.h>
#include <string>
#include <pthread.h>
#include <queue>
#include <semaphore.h>
#include <sys/time.h>
#include <signal.h>

#include "eventLoop.h"

using namespace std;



class deviceManager : public MyEventLoopThread {

public:
    deviceManager(const size_t deviceIndex);

protected:
    void InternalThreadEntry();


private:
    size_t tid;

};

#endif
