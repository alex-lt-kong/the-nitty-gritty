#ifndef PTHREAD_MUTEX_IN_SHM_COMMON_H
#define PTHREAD_MUTEX_IN_SHM_COMMON_H

#define SHM_SIZE (4096 + 128)
// man shm_open recommends to use a name starting with a slash and contains
// no other slashes.
#define SHM_NAME "/shm.me"
// o:wr, g:wr, i.e., 0660
#define SHM_PERMS (S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP)

#endif // PTHREAD_MUTEX_IN_SHM_COMMON_H
