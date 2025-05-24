#define _GNU_SOURCE
#include <limits.h>
#include <sched.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

volatile sig_atomic_t ev_flag = 0;

void signal_handler(const int signum) {
    char msg[] = "Signal [  ] caught\n";
    msg[8] = '0' + (char) (signum / 10);
    msg[9] = '0' + (char) (signum % 10);
    write(STDOUT_FILENO, msg, strlen(msg));
    ev_flag = 1;
}

int main(const int, char *argv[]) {
    int rc = 0;
    const int cpu_count = sysconf(_SC_NPROCESSORS_ONLN);
    struct sigaction act;
    act.sa_handler = signal_handler;
    sigemptyset(&act.sa_mask);
    act.sa_flags = SA_RESETHAND;
    if (sigaction(SIGINT, &act, 0) != 0 || sigaction(SIGTERM, &act, 0) != 0) {
        perror("sigaction()");
        rc = EXIT_FAILURE;
        goto err_sigaction;
    }
    srand(time(NULL));

    while (!ev_flag) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        const int cpu_id1 = rand() % cpu_count;
        const int cpu_id2 = rand() % cpu_count;
        CPU_SET(cpu_id1, &cpuset);
        CPU_SET(cpu_id2, &cpuset);

        if (sched_setaffinity(0, sizeof(cpuset), &cpuset) == -1) {
            perror("sched_setaffinity()");
            break;
        }
        printf("Process (PID: %d) now pinned to cores %d and %d\n", getpid(),
               cpu_id1, cpu_id2);
        const uint64_t iter_count =
                INT_MAX; // change this based on your CPU speed
        for (uint64_t i = 0; i < iter_count && !ev_flag; ++i) {
            const int r = rand();
            if (r == i) {
                printf("Optimization defeating message: %d from process (PID: "
                       "%d)\n",
                       r, getpid());
            }
        }
    }
err_sigaction:
    printf("%s exited gracefully\n", argv[0]);
    return rc;
}
