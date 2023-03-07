#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <signal.h>

int main(int argc, char** argv) {

    if (argc != 2) {
        printf("Usage: %s <0|1|2>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int pipefd_stdout[2], pipefd_stderr[2];
    // pipefd[0] is the read end of the pipe
    // pipefd[1] is the write end of the pipe
    FILE* stdout;
    FILE* stderr;
    char line[256];
    int status;

    if (pipe(pipefd_stdout) == -1 || pipe(pipefd_stderr) == -1) {
        // man 2 pipe
        perror("pipe()");
        return EXIT_FAILURE;
    }
    
    pid_t child_pid = fork(); //span a child process

    if (child_pid == -1) { // fork() failed, no child process created
        perror("fork(): ");
        return EXIT_FAILURE;
    }

    if (child_pid == 0) { // fork() succeeded, we are in the child process
        close(pipefd_stdout[0]);
        close(pipefd_stderr[0]);
        // man dup2
        dup2(pipefd_stdout[1], STDOUT_FILENO);
        dup2(pipefd_stderr[1], STDERR_FILENO);

        // Prepared three possible cases, to demo different behaviors
        if (atoi(argv[1]) == 0) {
            execl("./sub.out", "./sub.out", (char*) NULL); 
        } else if (atoi(argv[1]) == 1) {
            char *const parmList[] = {"/bin/ls", "-l", "/tmp/", NULL};
            execv("/bin/ls", parmList);
        } else {
            char *const parmList[] = {"/bin/ls", "-l", "/path/that/definitely/does/not/exist/", NULL};
            execv("/bin/ls", parmList);
        }

        perror("execl()");
        // The exec() functions return only if an error has occurred.
        // The return value is -1, and errno is set to indicate the error.
        exit(EXIT_FAILURE);
        // Have to exit() explicitly in case of execl() failure.
    }
    
    //Only parent gets here
    close(pipefd_stdout[1]);
    close(pipefd_stderr[1]);

    if ((stdout = fdopen(pipefd_stdout[0], "r")) == NULL) {
        perror("fdopen()");
        return EXIT_FAILURE;
    }
    if ((stderr = fdopen(pipefd_stderr[0], "r")) == NULL) {
        perror("fdopen()");
        fclose(stdout);
        return EXIT_FAILURE;
    }

    while(fgets(line, sizeof(line) - 1, stdout)) {            
        printf("stdout: %s", line);
    }
    while(fgets(line, sizeof(line) - 1, stderr)) {            
        printf("stderr: %s", line);
    }
    fclose(stdout);
    fclose(stderr);

    // wait for the child process to terminate
    waitpid(child_pid, &status, 0);
    if (WIFEXITED(status)) {
        printf("Child process exited normally, rc: %d\n", WEXITSTATUS(status));
    } else {
        printf("Child process exited abnormally.\n");
    }
    
}