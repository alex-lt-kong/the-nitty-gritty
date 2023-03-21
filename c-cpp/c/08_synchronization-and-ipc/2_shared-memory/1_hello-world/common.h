#define SHM_SIZE 512
// man shm_open recommends to use a name starting with a slash and contains
// no other slashes.
#define SHM_NAME "/shm.me"
// o:wr, g:wr, i.e., 0660
#define SHM_PERMS (S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP)
