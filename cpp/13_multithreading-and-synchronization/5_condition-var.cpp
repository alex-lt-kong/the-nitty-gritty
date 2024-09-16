#include <string>
#include <vector>
#include <iostream>
#include <thread>
#include <chrono>
#include <unistd.h>
#include <atomic>
#include <mutex>
#include <queue>
#include <condition_variable>

#include "common.h"

using namespace std;

static size_t num_objects = 10;

condition_variable cv;
queue<string> msg_queue;
mutex stdout_mutex;

/* While bool might be implemented in an atomic manner on x86/ARM/etc, 
this is not guaranteed by C++ standard. To play it safe and to be
more standard-compliant, we may want to use the more proper atomic<bool>
type*/
atomic<bool> should_stop;


void stdout_writer() {
    while (!should_stop) {
        unique_lock<mutex> lk(stdout_mutex);
        /* The 2nd parameter, a predicate which returns ​false if the waiting
        should be continued (bool(stop_waiting()) == false). The signature of
        the predicate function should be equivalent to the following:
        bool pred();​
        cv.wait() implicitly calls lk.unlock() at the moment it blocks this
        thread, this is also why it won't cause a deadlock when it is wait()ing
        */
        cv.wait(lk, []{ return !msg_queue.empty(); });
        string msg = msg_queue.front();
        msg_queue.pop();
        cout << msg << endl;
    }
    cout << "[stdout_writer()] exited gracefully" << endl;
}

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
        size_t count = 0;
        while (!should_stop) {
            this_thread::sleep_for(chrono::milliseconds(1000));
            // As lk's scope is this iteration, it will automatically unlock()
            // when it goes out of scope. This follows the paradigm of RAII
            string msg = "[" + to_string(thread_id) + "] iterating... " +
                to_string(count++);
            unique_lock<mutex> lk(stdout_mutex);
            msg_queue.push(msg);
            lk.unlock();
            /* There is an alternative method, cv.notiy_all(), but given
            that there is only one waiting thread, these two are equivalent.*/
            cv.notify_one();
            this_thread::sleep_for(chrono::milliseconds(thread_id * count));

        }
        cout << "[" << thread_id << "] event_loop() exited gracefully" << endl;
    }
    void wait() {
        if (th.joinable()) {
            th.join();
        }
    }

private:
    thread th;
};
vector<MyClass> objs;

int main() {
    should_stop = false;
    for (int i = 0; i < num_objects; ++i) {
        objs.push_back(MyClass(i));
    }

    cout << "Launched from the main\n";

    thread th = thread(&stdout_writer);
    /* Per C++ Standard Committee's design, a thread must be either join()ed
    or detach()ed, even if the exeuction already quits at the time of
    join()ing or detach()ing; otherwise it causes "terminate called
    without an active exception" then SIGABRT: 
    https://en.cppreference.com/w/cpp/thread/thread/~thread
    In C program, while I haven't seen this problem, some similar issues
    seem to show the same behavior.
    */
    th.detach();

    for (int i = 0; i < num_objects; ++i) {
        objs[i].start();
    }
    install_signal_handler();
    for (int i = 0; i < num_objects; ++i) {
        objs[i].wait();
    }
    return 0;
}
