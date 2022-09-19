#define SEM_INITIAL_VALUE 1
#define SEM_NAME "MySemaphore1"
#define SEM_PERMS (S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP)
// o:wr, g:wr, i.e., 0660

#define SHM_SIZE 65536
#define SHM_NAME "/shm.me"