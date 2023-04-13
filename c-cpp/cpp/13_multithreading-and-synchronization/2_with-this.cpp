#include <vector>
#include <iostream>
#include <thread>
#include <chrono>
#include <signal.h>
#include <unistd.h>

using namespace std;

static size_t num_objects = 10;

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
        running = true;
        th = thread(&MyClass::event_loop, this);
    }
    void event_loop() {        
        while (running) {
            this_thread::sleep_for(chrono::milliseconds(1000));
            cout << "[" << thread_id << "] iterating..." << endl; 
        }
        cout << "[" << thread_id << "] event_loop() exited gracefully" << endl;
    }
    void wait() {
        if (th.joinable()) {
            th.join();
        }
    }
    void stop() {
        running = false;
    }

private:
    thread th;
    bool running;
};
vector<MyClass> objs;

static void signal_handler(int signum) {
    char msg[] = "Signal [  ] caught\n";
    msg[8] = '0' + signum / 10;
    msg[9] = '0' + signum % 10;
    size_t len = sizeof(msg) - 1;
    size_t written = 0;
    while (written < len) {
        ssize_t ret = write(STDOUT_FILENO, msg + written, len - written);
        if (ret == -1) {
            perror("write()");
            break;
        }
        written += ret;
    }
    for (int i = 0; i < num_objects; ++i) {
        objs[i].stop();
    }
}

void install_signal_handler() {
    struct sigaction act;
    if (sigemptyset(&act.sa_mask) == -1) {
        perror("sigemptyset()");
        abort();
    }
    act.sa_handler = signal_handler;  
    act.sa_flags = 0;
    if (sigaction(SIGINT,  &act, 0) + sigaction(SIGABRT, &act, 0) +
        sigaction(SIGQUIT, &act, 0) + sigaction(SIGTERM, &act, 0) < 0) {
        perror("sigaction()");
        abort();
    }
}

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
