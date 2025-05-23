#include "pthread-mutex-in-shm-common.h"

#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

volatile sig_atomic_t ev_flag = 0;
pthread_mutex_t my_mutex;

void signal_handler(const int signum) {
  char msg[] = "Signal [  ] caught\n";
  msg[8] = '0' + (char)(signum / 10);
  msg[9] = '0' + (char)(signum % 10);
  write(STDOUT_FILENO, msg, strlen(msg));
  ev_flag = 1;
}

int main() {
  char buffer[SHM_SIZE];
  struct timespec ts;
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
  /* On Linux, one can check the status of shared memory items by
  ls -alh /dev/shm
  One may also notice that PERMS are not fully effective, we need to call
  umask() to make it work:
  https://stackoverflow.com/questions/51068208/shm-open-not-setting-group-write-access
  */
  const int fd = shm_open(SHM_NAME, O_RDWR | O_CREAT, SHM_PERMS);
  /* O_RDWR: open an existing shm for rw
     O_CREAT: create a new shm if it doesn't exist
     We should specify O_CREAT only on the writer side, or reader may
     inadvertently create a brand new shm object.
  */
  if (fd < 0) {
    perror("shm_open()");
    rc = -1;
    goto err_shm_open;
  }

  /* Despite the confusing name, it essentially means something
  similar to malloc() */
  if (ftruncate(fd, SHM_SIZE) == -1) {
    perror("ftruncate()");
    rc = -1;
    goto err_ftruncate;
  }

  /* get a pointer to memory */
  char *memptr =
      mmap(NULL, /* addr set to NULL so that the kernel chooses the address  */
           SHM_SIZE,               /* bytes since addr */
           PROT_READ | PROT_WRITE, /* access protections */
           MAP_SHARED,             /* mapping visible to other processes */
           fd,                     /* file descriptor */
           0 /* offset since fd: 0 to start from the beginning */
      );
  if (memptr == MAP_FAILED) {
    perror("mmap()");
    rc = -1;
    goto err_mmap;
  }

  printf("shm address: %p [0..%d]\n", memptr, SHM_SIZE - 1);
  while (!ev_flag) {
    timespec_get(&ts, TIME_UTC);
    memset(buffer, 0, SHM_SIZE);
    snprintf(buffer, SHM_SIZE - 1, "Shared memory content at %ld.%09ld\n",
             ts.tv_sec, ts.tv_nsec);
    printf("Written the below to shm:\n%s\n", buffer);
    strncpy(memptr, buffer, SHM_SIZE - 1);
    sleep(1);
  }

  /* clean up */
  /* unmap the storage */
  /* This is no C++ and no RAII, so nothing guarantees that we can't have
   error during finalization. */
  if (munmap(memptr, SHM_SIZE) != 0) {
    perror("munmap()");
  }
err_mmap:
err_ftruncate:
  /* On the writer side, we want to both shm_unlink() and close() the shared
  memory object. By shm_unlink()ing the shm object, we only remove the name
  of a referenced shm object. It is like a pointer goes out of scope--we
  can no longer use the pointer to refer to the shm object, but the object
  on heap won't be automatically gone. To get ride of the object,
  we need to close() it, just like we need to free() a pointer
  Note with a pointer, we must free() it once, or we are doomed. For
  close(), each process that shm_open()'s it should close() once, the
  underlying resources will be released when the last process close()'s it.*/
  if (shm_unlink(SHM_NAME) != 0) {
    /* We check all possible errors, but actually there is nothing we
    can do expect sending the error to stderr.*/
    perror("shm_unlink()");
  }
  if (close(fd) != 0) {
    perror("close()");
  }
err_shm_open:
err_sigaction:
  return rc;
}