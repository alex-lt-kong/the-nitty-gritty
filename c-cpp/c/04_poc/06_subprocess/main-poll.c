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


#define BUFSIZE 256

int exec(char** argv) {
    int pipefd_out[2], pipefd_err[2];
    // pipefd[0] is the read end of the pipe
    // pipefd[1] is the write end of the pipe
    FILE* fp_out;
    FILE* fp_err;
    char buff[BUFSIZE];
    int status;

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
        close(pipefd_out[0]);
        close(pipefd_err[0]);
        // man dup2
        dup2(pipefd_out[1], STDOUT_FILENO);
        dup2(pipefd_err[1], STDERR_FILENO);

        // Prepared a few possible cases, to demo different behaviors
        if (atoi(argv[1]) == 0) {
            execl("./sub.out", "./sub.out", (char*) NULL);
        } else if (atoi(argv[1]) == 1) {
            const char * args[] = {"./sub.out", "segfault", NULL};
            execv(args[0], args);
        } else if (atoi(argv[1]) == 2) {
            const char * args[] = {"./sub.out", "flooding", NULL};
            execv(args[0], args);
        } else if (atoi(argv[1]) == 3) {
            const char * args[] = {"/bin/ls", "-l", "/tmp/", NULL};
            execv(args[0], args);
        } else {
            
            const char * args[] = {"/bin/ls", "-l",
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


    fd_set rfds;
    int retval;

    /* Watch stdin (fd 0), stdout (fd 1) and stderr (fd 2) to see when it has I/O. */
    FD_ZERO(&rfds);
    FD_SET(pipefd_out[0], &rfds);
    FD_SET(pipefd_err[0], &rfds);


    int nfds, num_open_fds;
    num_open_fds = nfds = 2;
    struct pollfd pfds[2];

    /* Open each file on command line, and add it 'pfds' array. */


    pfds[0].fd = pipefd_out[0];
    pfds[0].events = POLLIN;
    pfds[1].fd = pipefd_err[0];
    pfds[1].events = POLLIN;

    /* Keep calling poll() as long as at least one file descriptor is
       open. */

    while (num_open_fds > 0) {
        int ready;

        printf("About to poll()\n");
        ready = poll(pfds, nfds, -1);
        if (ready == -1)
            perror("poll()");

        printf("Ready: %d\n", ready);

        /* Deal with array returned by poll(). */

        for (int j = 0; j < nfds; j++) {
            char buf[4096];

            if (pfds[j].revents != 0) {
                printf("  fd=%d; events: %s%s%s\n", pfds[j].fd,
                        (pfds[j].revents & POLLIN)  ? "POLLIN "  : "",
                        (pfds[j].revents & POLLHUP) ? "POLLHUP " : "",
                        (pfds[j].revents & POLLERR) ? "POLLERR " : "");

                if (pfds[j].revents & POLLIN) {
                    ssize_t s = read(pfds[j].fd, buf, sizeof(buf));
                    if (s == -1)
                        perror("read");
                    printf("    read %zd bytes: %.*s\n", s, (int) s, buf);
                } else {                /* POLLERR | POLLHUP */
                    printf("    closing fd %d\n", pfds[j].fd);
                    if (close(pfds[j].fd) == -1)
                        perror("close");
                    num_open_fds--;
                }
            }
        }
    }

    printf("All file descriptors closed; bye\n");







/*

    if ((fp_out = fdopen(pipefd_out[0], "r")) == NULL) {
        perror("fdopen()");
        goto err_err_fds;
    }
    if ((fp_err = fdopen(pipefd_out[0], "r")) == NULL) {
        perror("fdopen()");
        goto err_out_file;
    }
    printf("===== stdout =====\n");
    while(fgets(buff, sizeof(buff) - 1, fp_out)) {
        printf("%s", buff);
    }
    printf("===== stdout =====\n");

    printf("===== stderr =====\n");
    while(fgets(buff, sizeof(buff) - 1, fp_err)) {
        printf("%s", buff);
    }    
    printf("===== stderr =====\n");
*/
/*
    if (fclose(fp_out) != 0) { perror("fclose(fp_out)"); }
    if (fclose(fp_err) != 0) { perror("fclose(fp_err)"); }
    if (close(pipefd_out[0]) != 0) { perror("close(pipefd_out)"); }
    if (close(pipefd_err[0]) != 0) { perror("close(pipefd_err)"); }*/

    // wait for the child process to terminate
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


err_out_file:
        fclose(fp_out);
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
        printf("Usage: %s <0|1|2|3>\n", argv[0]);
        return EXIT_FAILURE;
    }
    return exec(argv);    
}