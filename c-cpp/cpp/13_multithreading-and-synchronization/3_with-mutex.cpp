#include <vector>
#include <iostream>
#include <thread>
#include <chrono>
#include <signal.h>
#include <unistd.h>
#include <atomic>
#include <mutex>

#include "common.h"

using namespace std;

static size_t num_objects = 10;
mutex stdout_mutex;
/* While bool might be implemented in an atomic manner on x86/ARM/etc, 
this is not guaranteed by C++ standard. To play it safe and to be
more standard-compliant, we may want to use the more proper atomic<bool>
type*/
atomic<bool> should_stop;

class MyClass {

public:
    int thread_id;
    MyClass(int thread_id) {
        this->thread_id = thread_id;
    }
    MyClass(const MyClass& rhs) {
        this->thread_id = rhs.thread_id;
    }
    ~MyClass() {}
    void start() {
        th = thread(&MyClass::event_loop, this);
    }
    void event_loop() {        
        while (!should_stop) {
            this_thread::sleep_for(chrono::milliseconds(1000));
            unique_lock<mutex> lk(stdout_mutex);
            // As lk's scope is this iteration, it will automatically unlock()
            // when it goes out of scope. This implements the RAII pattern.
            cout << "[" << thread_id << "] iterating..." << endl;
        }
        cout << "[" << thread_id << "] event_loop() exited gracefully" << endl;
    }
    void wait() {
        // joinable() is a must--sometimes a thread already quits when we want 
        // to join(), causing unexpected behavior
        if (th.joinable()) {
            th.join();
        }
    }

private:
    thread th;
};
vector<MyClass> objs;


int main() {

    for (int i = 0; i < num_objects; ++i) {
        objs.push_back(MyClass(i));
    }

    cout << "Launched from the main\n";

    for (int i = 0; i < num_objects; ++i) {
        objs[i].start();
    }
    install_signal_handler();
    for (int i = 0; i < num_objects; ++i) {
        objs[i].wait();
    }

    return 0;
}
