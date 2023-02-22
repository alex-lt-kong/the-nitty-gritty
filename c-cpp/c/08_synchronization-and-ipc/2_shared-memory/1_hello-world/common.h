#define SHM_SIZE 512
#define SHM_NAME "/shm.me"
#define SHM_PERMS (S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP)
// o:wr, g:wr, i.e., 0660