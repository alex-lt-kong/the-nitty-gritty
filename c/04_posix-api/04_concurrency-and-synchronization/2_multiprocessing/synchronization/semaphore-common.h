#pragma once

#include <signal.h>
#include <unistd.h>

#define INITIAL_VALUE 1
#define SEM_NAME "MySemaphore0"
#define SEM_PERMS (S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP)
