#define SEM_INITIAL_VALUE 1
#define SEM_NAME "MySemaphore5"

#define SHM_SIZE 4096
// The name of a shm object should start with a slash (/).
#define SHM_NAME "/shm.me"

#define PERMS (S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP)
// o:wr, g:wr, i.e., 0660

struct timespec ts;
