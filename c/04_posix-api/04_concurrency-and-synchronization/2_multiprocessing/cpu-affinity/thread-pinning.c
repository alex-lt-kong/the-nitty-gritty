#define _GNU_SOURCE // Needed for non-POSIX API pthread_setaffinity_np()
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

volatile sig_atomic_t ev_flag = 0;

pid_t get_tid() {
    // Linux-specific way to get Thread ID (TID)
    return (pid_t) syscall(SYS_gettid);
}

void signal_handler(const int signum) {
    char msg[] = "Signal [  ] caught\n";
    msg[8] = '0' + (char) (signum / 10);
    msg[9] = '0' + (char) (signum % 10);
    write(STDOUT_FILENO, msg, strlen(msg));
    ev_flag = 1;
}

void *thread_func(void *arg) {

    const int cpu_count = sysconf(_SC_NPROCESSORS_ONLN);
    const int thread_num = *(int *) arg;
    int cpu_id = thread_num;
    while (!ev_flag) {
        cpu_id = rand() % cpu_count;
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpu_id, &cpuset);
        const pthread_t thread = pthread_self();

        if (pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset) != 0) {
            perror("pthread_setaffinity_np()");
            exit(EXIT_FAILURE);
        }
        // Verify the thread is running on the correct CPU
        if (pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset) != 0) {
            perror("pthread_getaffinity_np");
            exit(EXIT_FAILURE);
        }

        printf("Thread (PID: %d, TID: %d) running on CPU %d\n", getpid(),
               get_tid(), sched_getcpu());
        const int iter_count = 200000000; // change this based on your CPU speed
        for (int i = 0; i < iter_count && !ev_flag; ++i) {
            const int r = rand();
            if (r == i) {
                printf("Optimization defeating message: %d from thread %d\n", r,
                       get_tid());
            }
        }
    }
    return NULL;
}

int main(const int, char *argv[]) {
    int rc = 0;
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

    pthread_t thread1, thread2;
    int thread1_core = 0;
    int thread2_core = 1;

    if ((rc = pthread_create(&thread1, NULL, thread_func, &thread1_core)) !=
        0) {
        fprintf(stderr, "pthread_create() failed: %s\n", strerror(rc));
        goto err_pthread_create1;
    }
    if ((rc = pthread_create(&thread2, NULL, thread_func, &thread2_core)) !=
        0) {
        fprintf(stderr, "pthread_create() failed: %s\n", strerror(rc));
        goto err_pthread_create2;
    }

    pthread_join(thread2, NULL);
err_pthread_create2:
    pthread_join(thread1, NULL);
err_pthread_create1:
err_sigaction:
    printf("%s exited gracefully\n", argv[0]);
    return rc;
}
