#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/wait.h>

#include "common.h"   

using namespace std;
using namespace cv;                                                          

volatile sig_atomic_t ev_flag = 0;

static void signal_handler(int signum) {
    ev_flag = 1;
    char msg[] = "Signal [  ] caught\n";
    msg[8] = '0' + signum / 10;
    msg[9] = '0' + signum % 10;    
    size_t len = sizeof(msg) - 1;
    size_t written = 0;
    while (written < len) {
        ssize_t ret = write(STDOUT_FILENO, msg + written, len - written);
        if (ret == -1 && errno == EINTR) {
            continue;
        }
        if (ret == -1) {
            perror("write()");
            break;
        }
        written += ret;
    }
}

void install_signal_handler() {
    if (_NSIG > 99) {
        fprintf(stderr, "Current design can't handle more than 99 signals\n");
        abort();
    }
    struct sigaction act;
    // Initialize the signal set to empty, similar to memset(0)
    if (sigemptyset(&act.sa_mask) == -1) {
        perror("sigemptyset()");
        abort();
    }
    act.sa_handler = signal_handler;
    /* SA_RESETHAND means we want our signal_handler() to intercept the signal
    once. If a signal is sent twice, the default signal handler will be used
    again. `man sigaction` describes more possible sa_flags. */
    act.sa_flags = SA_RESETHAND;
    
    if (sigaction(SIGINT, &act, 0)  + sigaction(SIGABRT, &act, 0) +
        sigaction(SIGQUIT, &act, 0) + sigaction(SIGTERM, &act, 0) +
        sigaction(SIGPIPE, &act, 0) + sigaction(SIGCHLD, &act, 0) +
        sigaction(SIGSEGV, &act, 0) + sigaction(SIGTRAP, &act, 0) < 0) {
        /* Could miss some error if more than one sigaction() fails. However,
        given that the program will quit if one sigaction() fails, this
        is not considered an issue */
        perror("sigaction()");
        abort();
    }
}


int main() {
    install_signal_handler();
    VideoCapture cap = openVideoSource();

    FILE *child_proc;
    child_proc = popen(ffmpegCommand.c_str(), "w");
    if (child_proc == NULL) {
        perror("popen()");
        return EXIT_FAILURE;
    }
    Mat frame;
    while (cap.read(frame) && ev_flag == 0) {
        size_t frameSize = frame.dataend - frame.datastart;
        if (fwrite(frame.data, 1, frameSize, child_proc) != frameSize) {
            perror("fwrite()");
            break;
        }
        
    }
    cout << "\nThe evloop quitted, gracefully\n" << endl;
    pclose(child_proc);                                                                      
    return 0;                                                                
} 
