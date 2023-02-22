#include "common.h"

int main() {
 
  memset(&receiver_sockaddr, 0, sizeof(struct sockaddr_un));
  memset(&sender_sockaddr, 0, sizeof(struct sockaddr_un));

	if ((fd = socket(PF_UNIX, SOCK_DGRAM, 0)) < 0) {
		perror("socket()");
    return 1;
	}
	
  sender_sockaddr.sun_family = AF_UNIX;
  strcpy(sender_sockaddr.sun_path, SENDER_SOCK_FILE);
  unlink(SENDER_SOCK_FILE); // unlink() so that we can bind() the file
  if (bind(fd, (struct sockaddr *)&sender_sockaddr, sizeof(sender_sockaddr)) < 0) {
    perror("bind()");
    return 2;
  }	
	
  memset(&receiver_sockaddr, 0, sizeof(receiver_sockaddr));
  receiver_sockaddr.sun_family = AF_UNIX;
  strcpy(receiver_sockaddr.sun_path, RECEIVER_SOCK_FILE);
  if (connect(fd, (struct sockaddr *)&receiver_sockaddr, sizeof(receiver_sockaddr)) == -1) {
    perror("connect()");
    return 3;
  }
  char time_str[64];
  timespec_get(&ts, TIME_UTC);
  snprintf(time_str, 64, "%ld.%09ld", ts.tv_sec, ts.tv_nsec);
  strcpy(buff, time_str);
  memset(buff + strlen(time_str), 'A', MSG_BUF_SIZE - (strlen(time_str) + 1));
  timespec_get(&ts, TIME_UTC);
  snprintf(time_str, 64, "%ld.%09ld", ts.tv_sec, ts.tv_nsec);
  strcpy(buff + MSG_BUF_SIZE - (strlen(time_str) + 1), time_str);
  if ((len = send(fd, buff, strlen(buff)+1, 0)) == -1) {
    perror("send()");
    return 4;
  }
  printf ("send()'ed %ld bytes\n", len);

  if ((len = recv(fd, buff, MSG_BUF_SIZE, 0)) < 0) {
    perror("recv");
    return 5;
  }
  printf ("recv()'ed %ld bytes: [%s]\n", len, buff);


	if (fd >= 0) {
		close(fd);
	}

	unlink(SENDER_SOCK_FILE);
	return 0;
}