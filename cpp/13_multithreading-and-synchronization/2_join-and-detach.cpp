#include <iostream>
#include <thread>

static const int num_threads = 10;

using namespace std;

void call_from_thread(int tid) {
    this_thread::sleep_for(chrono::milliseconds(1000));
    cout << "Launched by thread [" << tid << "]" << endl;
    this_thread::sleep_for(chrono::milliseconds(5000));
}

int main() {
    
    thread th;    
    cout << "before 1st start, joinable?: " << th.joinable() << '\n';
    th = thread(call_from_thread, 0);
    cout << "after 1st start, joinable?: " << th.joinable() << '\n';
    th.join();
    cout << "after join(), joinable?: " << th.joinable() << '\n';
 
    th = thread(call_from_thread, 1);
    cout << "after 2nd start, joinable?: " << th.joinable() << '\n';
    th.detach();
    cout << "after detach(), joinable?: " << th.joinable() << '\n';
    cout << "main() returned" << endl;
    // upon return, the thread will exit immediately, we won't see
    // a 2nd Launched by thread [1] message.
    return 0;
}