#ifndef DEVICE_MANAGER_H
#define DEVICE_MANAGER_H

#include <string>
#include <pthread.h>


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
