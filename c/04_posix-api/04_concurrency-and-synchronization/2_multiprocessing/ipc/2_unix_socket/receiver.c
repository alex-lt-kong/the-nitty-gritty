#include "common.h"


int main() {
	int ret;
  memset(&receiver_sockaddr, 0, sizeof(receiver_sockaddr));
  memset(&sender_sockaddr, 0, sizeof(sender_sockaddr));
	socklen_t fromlen = sizeof(sender_sockaddr);

	if ((fd = socket(PF_UNIX, SOCK_DGRAM, 0)) < 0) {
		perror("socket()");
		return 1;
	}
	
  
  /***************************************/
  /* Set up the UNIX sockaddr structure  */
  /* by using AF_UNIX for the family and */
  /* giving it a filepath to bind to.    */
  /*                                     */
  /* Unlink the file so the bind will    */
  /* succeed, then bind to that file.    */
  /***************************************/
  
  receiver_sockaddr.sun_family = AF_UNIX;
  strcpy(receiver_sockaddr.sun_path, RECEIVER_SOCK_FILE);
  unlink(RECEIVER_SOCK_FILE);
  if (bind(fd, (struct sockaddr *)&receiver_sockaddr, sizeof(receiver_sockaddr)) < 0) {
    perror("bind()");
    return 1;
  }
	

	while ((len = recvfrom(fd, buff, MSG_BUF_SIZE, 0, (struct sockaddr *)&sender_sockaddr, &fromlen)) > 0) {
    timespec_get(&ts, TIME_UTC);
		printf("recvfrom: [%s] at %ld.%09ld\n", buff, ts.tv_sec, ts.tv_nsec);
		strcpy(buff, "transmit good!");
		ret = sendto(fd, buff, strlen(buff)+1, 0, (struct sockaddr *)&sender_sockaddr, fromlen);
		if (ret < 0) {
			perror("sendto");
			break;
		}
	}
	

	if (fd >= 0) {
		close(fd);
	}

	return 0;
}