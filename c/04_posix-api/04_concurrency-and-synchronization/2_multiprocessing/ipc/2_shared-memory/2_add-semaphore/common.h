#define SEM_INITIAL_VALUE 1
/* man sem_overview: A named semaphore is identified by a name of the form
/somename; that is, a null-terminated string of up to NAME_MAX-4 (i.e., 251)
characters consisting of an initial slash, followed by one or more  
characters, none of which are slashes.*/
#define SEM_NAME "/my.sem"

#define SHM_SIZE 4096
// The name of a shm object should start with a slash (/).
#define SHM_NAME "/shm.me"

#define PERMS (S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP)
// o:wr, g:wr, i.e., 0660

struct timespec ts;
