// https://babbage.cs.qc.cuny.edu/courses/cs701/2003_02/final.cc.html

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <signal.h>
#include <sys/select.h>
#include <poll.h>
#include <stdint.h>
#include <fcntl.h>
#include <time.h>

//#define ENABLE_TIMEOUT

#define BUFSIZE 1024

#ifdef ENABLE_TIMEOUT
struct timespec start, now;
long elapsed_usecs = 0;
long timeout_usecs = 10 * 1000000;// 10 sec
#endif

int exec(char* argv1) {
    int pipefd_out[2], pipefd_err[2];
    // pipefd[0] is the read end of the pipe
    // pipefd[1] is the write end of the pipe


    if (pipe(pipefd_out) == -1) {
        // man 2 pipe
        perror("pipe()");
        goto err_initial;
    }
    if (pipe(pipefd_err) == -1) {
        perror("pipe()");
        goto err_out_fds;
    }
    
    pid_t child_pid = fork(); //spawn a child process

    if (child_pid == -1) { // fork() failed, no child process created
        perror("fork()");
        goto err_err_fds;
    }

#ifdef ENABLE_TIMEOUT
    clock_gettime(CLOCK_MONOTONIC, &start);
    /*
    CLOCK_MONOTONIC: represents the absolute elapsed wall-clock time since
    some arbitrary, fixed point in the past. It isn't affected by changes in
    the system time-of-day clock.
    CLOCK_REALTIME: represents the machine's best-guess as to the current
    wall-clock, time-of-day time. As Ignacio and MarkR say, this means that
    CLOCK_REALTIME can jump forwards and backwards as the system time-of-day
    clock is changed, including by NTP.
    */
#endif

    if (child_pid == 0) { // fork() succeeded, we are in the child process
        if (close(pipefd_out[0]) == -1) { perror("close(pipefd_out[0])"); }
        if (close(pipefd_err[0]) == -1) { perror("close(pipefd_err[0])"); }
        if (dup2(pipefd_out[1], STDOUT_FILENO) == -1) {
            perror("close(pipefd_out[1])");
        }
        if (dup2(pipefd_err[1], STDERR_FILENO) == -1) {
            perror("close(pipefd_err[1])");
        }
        // Prepared a few possible cases, to demo different behaviors
        if (atoi(argv1) == 0) {
            execl("./sub.out", "./sub.out", NULL);
        } else if (atoi(argv1) == 1) {
            const char* args[] = {"./sub.out", "segfault", NULL};
            execv(args[0], args);
        } else if (atoi(argv1) == 2) {
            const char* args[] = {"./sub.out", "flooding", NULL};
            execv(args[0], args);
        } else if (atoi(argv1) == 3) {
            const char* args[] = {"./sub.out", "sleep", "16", NULL};
            execv(args[0], args);
        } else if (atoi(argv1) == 4) {
            const char* args[] = {"./sub.out", "sleep", "4", NULL};
            execv(args[0], args);    
        } else if (atoi(argv1) == 5) {
            const char* args[] = {"/bin/ls", "-l", "/tmp/", NULL};
            execv(args[0], args);
        } else {            
            const char* args[] = {"/bin/ls", "-l",
                "/path/that/definitely/does/not/exist/", NULL};
            execv(args[0], args);
        }

        perror("execl()/execv()");
        // The exec() functions return only if an error has occurred.
        // The return value is -1, and errno is set to indicate the error.
        _exit(EXIT_FAILURE);
        /* Have to _exit() explicitly in case of execl() failure.
           Difference between _exit() and exit()?
           
           The basic difference between exit() and _exit() is that the former
           performs clean-up related to user-mode constructs in the library,
           and calls user-supplied cleanup functions, whereas the latter
           performs only the kernel cleanup for the process.
           
           In the child branch of a fork(), it is normally incorrect to use
           exit(), because that can lead to stdio buffers being flushed
           twice, and temporary files being unexpectedly removed.
        */
    }
    
    //Only parent gets here
    if (close(pipefd_out[1]) == -1) { perror("close(ipefd_out[1])"); }
    if (close(pipefd_err[1]) == -1) { perror("close(pipefd_err[1])"); }

    struct pollfd pfds[] = {
        { pipefd_out[0], POLLIN, 0 },
        { pipefd_err[0], POLLIN, 0 },
    };
    nfds_t nfds = sizeof(pfds) / sizeof(struct pollfd);    
    int num_open_fds = (int)nfds;
    // num_open_fds must be int, not unsigned int; otherwise we risk
    // "underflow" it
    
    /* Keep calling poll() as long as at least one file descriptor is
       open. */
    while (num_open_fds > 0) {

#ifdef ENABLE_TIMEOUT
        if (elapsed_usecs >= timeout_usecs) {
            break;
        }
        clock_gettime(CLOCK_MONOTONIC, &now);
        elapsed_usecs = (now.tv_sec - start.tv_sec) * 1000000 +
                        (now.tv_nsec - start.tv_nsec) / 1000;
#endif
        int ready = poll(pfds, nfds, timeout_usecs / 1000);
        if (ready == -1)
            perror("poll()");

        /* Deal with array returned by poll(). */
        for (size_t j = 0; j < nfds; j++) {
            
            if (pfds[j].revents != 0) {
                /* If this buffer is too small and the child process prints
                   too fast, we can still saturate the buffer even if we
                   use poll()...*/
                char buf[4096] = {0};
                if (pfds[j].revents & POLLIN) {
                    ssize_t s = read(pfds[j].fd, buf, sizeof(buf)-1);
                    if (s == -1)
                        perror("read()");
                    if (j == 0) { printf("<stdout>%s</stdout>\n", buf); }
                    else { printf("<stderr>%s</stderr>\n", buf); }
                    fflush(stdout);
                } else {                /* POLLERR | POLLHUP */
                    num_open_fds--;
                }
            }
        }
    }

    if (close(pipefd_out[0]) == -1) { perror("close(ipefd_out[0])"); }
    if (close(pipefd_err[0]) == -1) { perror("close(pipefd_err[0])"); }
    
    // wait for the child process to terminate
    int status;
#ifdef ENABLE_TIMEOUT
    
    __useconds_t sleep_us = 1;
    while(waitpid(child_pid, &status, WNOHANG) == 0) {
        
        usleep(sleep_us);
        sleep_us = sleep_us >= 1000000 ? sleep_us : sleep_us * 2;
        clock_gettime(CLOCK_MONOTONIC, &now);
        elapsed_usecs = (now.tv_sec - start.tv_sec) * 1000000 +
                        (now.tv_nsec - start.tv_nsec) / 1000;
        if (elapsed_usecs > timeout_usecs) {
            printf("Timeout %lu ms reached, kill()ing process %d...\n",
                timeout_usecs / 1000, child_pid);
            // This avoid leaving a zombie process in the process table:
            // https://stackoverflow.com/questions/69509427/kill-child-process-spawned-with-execl-without-making-it-zombie
            // Its implication needs further research though.
            signal(SIGCHLD, SIG_IGN);
            if (kill(child_pid, SIGTERM) == -1) {
                perror("kill(child_pid, SIGTERM)");
                if (kill(child_pid, SIGKILL) == -1) {
                    perror("kill(child_pid, SIGKILL)");
                } else {
                    printf("kill()ed successfully with SIGKILL\n");
                    break;
                }
            } else {
                printf("kill()ed successfully\n");
                break; // Without break the while() loop could be entered again.
            }
        }
    }
#else
    if (waitpid(child_pid, &status, 0) == -1) {
        perror("waitpid()");
        return EXIT_FAILURE;
    }
#endif
    if (WIFEXITED(status)) {
        printf("Child process exited normally, rc: %d\n", WEXITSTATUS(status));
    } else {
        printf("Child process exited unexpectedly ");
        if (WIFSIGNALED(status)) {
            printf("(terminated by a signal: %d)\n", WTERMSIG(status));
        } else if (WIFSTOPPED(status)) {
            printf("(stopped by delivery of a signal: %d)\n", WSTOPSIG(status));
        } else {
            printf("(unknown status: %d)\n", status);
        }
    }
    return EXIT_SUCCESS;

err_err_fds:
        if (close(pipefd_err[0]) == -1) { perror("close(pipefd_err[0])"); }
        if (close(pipefd_err[1]) == -1) { perror("close(pipefd_err[1])"); }
err_out_fds:
        if (close(pipefd_out[0]) == -1) { perror("close(ipefd_out[0])"); }
        if (close(pipefd_out[1]) == -1) { perror("close(ipefd_out[1])"); }
err_initial:
        return EXIT_FAILURE;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage: %s <0|1|2|3|4|5|6>\n", argv[0]);
        return EXIT_FAILURE;
    }
    const size_t iter_count = atoi(argv[1]) == 4 ? 16 : 1;
    for (size_t i = 0; i < iter_count; ++i) {
        exec(argv[1]);
    }
    return 0;
}