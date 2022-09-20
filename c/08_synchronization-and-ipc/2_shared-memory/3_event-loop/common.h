#define SEM_INITIAL_VALUE 1
#define SEM_NAME "MySemaphore5"

#define SHM_SIZE 65535
#define SHM_NAME "/shm.me"

#define PERMS (S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP)
// o:wr, g:wr, i.e., 0660