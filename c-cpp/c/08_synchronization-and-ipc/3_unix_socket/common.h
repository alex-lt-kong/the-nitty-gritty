#ifndef IPC_H
#define IPC_H

#include <stdio.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#define SENDER_SOCK_FILE "/tmp/my-sender.sock"
#define RECEIVER_SOCK_FILE "/tmp/my-receiver.sock"

#define MSG_BUF_SIZE 65536

struct timespec ts;
char buff[MSG_BUF_SIZE];
struct sockaddr_un receiver_sockaddr; 
struct sockaddr_un sender_sockaddr;
size_t len;
int fd;
#endif