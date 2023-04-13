#include <iostream>
#include <thread>

static const int num_threads = 10;

using namespace std;

void call_from_thread(int tid) {
    cout << "Launched by thread " << tid << endl;
}

int main() {
    thread t[num_threads];

    for (int i = 0; i < num_threads; ++i) {
        t[i] = thread(call_from_thread, i);
    }

    cout << "Launched from the main\n";

    for (int i = 0; i < num_threads; ++i) {
        t[i].join();
    }

    return 0;
}