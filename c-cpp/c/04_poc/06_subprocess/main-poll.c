// https://babbage.cs.qc.cuny.edu/courses/cs701/2003_02/final.cc.html

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <signal.h>
#include <sys/select.h>
#include <poll.h>
#include <fcntl.h>


#define BUFSIZE 1024

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
    
    pid_t child_pid = fork(); //span a child process

    if (child_pid == -1) { // fork() failed, no child process created
        perror("fork()");
        goto err_err_fds;
    }

    if (child_pid == 0) { // fork() succeeded, we are in the child process
        if (close(pipefd_out[0]) == -1) { perror("close(pipefd_out[0])"); }
        if (close(pipefd_err[0]) == -1) { perror("close(pipefd_err[0])"); }
        if (dup2(pipefd_out[1], STDOUT_FILENO) == -1) {
            perror("close(pipefd_out[1]) in child");
        }
        if (dup2(pipefd_err[1], STDERR_FILENO) == -1) {
            perror("close(pipefd_err[1]) in child");
        }

        // Prepared a few possible cases, to demo different behaviors
        if (atoi(argv1) == 0) {
            execl("./sub.out", "./sub.out", NULL);
        } else if (atoi(argv1) == 1) {
            const char *const  args[] = {"./sub.out", "segfault", NULL};
            execv(args[0], args);
        } else if (atoi(argv1) == 2) {
            const char *const  args[] = {"./sub.out", "flooding", NULL};
            execv(args[0], args);
        } else if (atoi(argv1) == 3) {
            const char *const  args[] = {"/bin/ls", "-l", "/tmp/", NULL};
            execv(args[0], args);
        } else {            
            const char *const  args[] = {"/bin/ls", "-l",
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
    close(pipefd_out[1]);
    close(pipefd_err[1]);

    struct pollfd pfds[] = {
        { pipefd_out[0], POLLIN, 0 },
        { pipefd_err[0], POLLIN, 0 },
    };
    int nfds = sizeof(pfds) / sizeof(struct pollfd);    
    int num_open_fds = nfds;
    
    /* Keep calling poll() as long as at least one file descriptor is
       open. */
    while (num_open_fds > 0) {
        int ready = poll(pfds, nfds, -1);
        if (ready == -1)
            perror("poll()");

        /* Deal with array returned by poll(). */
        for (int j = 0; j < nfds; j++) {
            /* If this buffer is too slow and the child process prints
               too fast, we can still saturate the buffer...*/
            
            if (pfds[j].revents != 0) {
                char buf[4096] = {0};
                if (pfds[j].revents & POLLIN) {
                    ssize_t s = read(pfds[j].fd, buf, sizeof(buf)-1);
                    if (s == -1)
                        perror("read()");
                    if (j == 0) { printf("<stdout>%s</stdout>\n", buf); }
                    else { printf("<stderr>%s</stderr>\n", buf); }
                    fflush(stdout);
                } else {                /* POLLERR | POLLHUP */
                    if (close(pfds[j].fd) == -1)
                        perror("close");
                    num_open_fds--;
                }
            }
        }
    }

    // wait for the child process to terminate
    int status;
    if (waitpid(child_pid, &status, 0) == -1) {
        perror("waitpid()");
        return EXIT_FAILURE;
    }
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
        close(pipefd_err[0]);
        close(pipefd_err[1]);
err_out_fds:
        close(pipefd_out[0]);
        close(pipefd_out[1]);
err_initial:
        return EXIT_FAILURE;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage: %s <0|1|2|3|4>\n", argv[0]);
        return EXIT_FAILURE;
    }
    return exec(argv[1]);    
}