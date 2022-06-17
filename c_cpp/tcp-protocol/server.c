#include <stdio.h>
#include <sys/socket.h>
#include <unistd.h>
#include <stdlib.h>
#include <netinet/in.h>
#include <string.h>

int main() {
    int server_fd; // _fd should stand for file descriptor.
    int new_socket;
    long valread;
    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    /*
     * domain: AF_INET means the IP address family
     * type:...
     * protocol: For TCP/IP sockets, there’s only one form of virtual circuit service, so the last argument, protocol, is zero.
     */
    if (server_fd < 0) {
        perror("cannot create socket");
        return 1; 
    }

    /* Bing a socket */
    struct sockaddr_in address;
    /* Because sockets were designed to work with various different types of communication interfaces, the interface
     * is very general. Instead of accepting, say, a port number as a parameter, it takes a sockaddr structure whose
     * actual format is determined on the address family (type of network) you're using.
     * 
     * struct sockaddr_in 
     * { 
     *     __uint8_t         sin_len;  // The address family we used when we set up the socket. In our case, it’s AF_INET.
     *     sa_family_t       sin_family; 
     *     in_port_t         sin_port; 
     *     struct in_addr    sin_addr; 
     *     char              sin_zero[8]; 
     * };
     */
    int addrlen = sizeof(address);
    char *hello = "Hello from server";
    const int PORT = 8080; //Where the clients can reach at

    /* htonl converts a long integer (e.g. address) to a network representation */
    /* htons converts a short integer (e.g. port) to a network representation */ 
    memset((char*)&address, 0, sizeof(address)); 
    address.sin_family = AF_INET; 
    address.sin_addr.s_addr = htonl(INADDR_ANY);  // It does NOT "generate a random IP". It binds the socket to all available interfaces, i.e., binds to 0.0.0.0.
    address.sin_port = htons(PORT);
    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) { 
        /* To bind/name a socket, means assigning a transport address to the socket. In the case of IP networking, it
         * is a port number. In sockets, this operation is called binding an address.
         */
        perror("bind faileding"); 
        return 0; 
    }

    if (listen(server_fd, 10) < 0) {
        /* listen() marks the socket referred to by sockfd as a passive socket, that is, as a socket that will be
         * used to accept incoming connection requests using accept().
         * sockfd: a file descriptor that refers to a socket of type SOCK_STREAM or SOCK_SEQPACKET.
         * backlog, defines the maximum number of pending connections that can be queued up before connections are refused.
        */
        perror("In listen");
        exit(EXIT_FAILURE);
    }
    while(1) {
        printf("\n+++++++ Waiting for new connection ++++++++\n\n");
        if ((new_socket = accept(server_fd, (struct sockaddr*)&address, (socklen_t*)&addrlen))<0) {
            /* The accept system call grabs the first connection request on the queue of pending connections
             *  (set up in listen) and creates a new socket for that connection.
             * The original socket that was set up for listening is used only for accepting connections,
             * not for exchanging data. By default, socket operations are synchronous, and accept will block until
             * a connection is present on the queue.
             */
            perror("In accept");
            exit(EXIT_FAILURE);
        }
        
        char buffer[30000] = {0};
        valread = read(new_socket, buffer, 30000);
        printf("%s\n", buffer);
        write(new_socket, hello ,strlen(hello));
        printf("------------------Hello message sent-------------------\n");
        close(new_socket);
    }
    return 0;
}