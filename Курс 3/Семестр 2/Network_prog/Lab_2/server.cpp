#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

// Максимальное число ожидающих соединений
#define BACKLOG 5

void sigchld_handler(int signo)
{
    while (waitpid(-1, NULL, WNOHANG) > 0)
        ;
}

int main(void)
{
    int sockfd, new_fd;
    struct sockaddr_in server_addr, client_addr;
    socklen_t sin_size;
    int on = 1;

    if ((sockfd = socket(AF_INET, SOCK_STREAM, 0)) == -1) {
        perror("socket");
        exit(EXIT_FAILURE);
    }

    if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(int)) == -1) {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }

    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = 0;

    if (bind(sockfd, (struct sockaddr*)&server_addr, sizeof(server_addr)) == -1) {
        perror("bind");
        exit(EXIT_FAILURE);
    }

    socklen_t addr_len = sizeof(server_addr);
    if (getsockname(sockfd, (struct sockaddr*)&server_addr, &addr_len) == -1) {
        perror("getsockname");
        exit(EXIT_FAILURE);
    }
    printf("Server listening on port %d\n", ntohs(server_addr.sin_port));

    if (listen(sockfd, BACKLOG) == -1) {
        perror("listen");
        exit(EXIT_FAILURE);
    }

    struct sigaction sa;
    sa.sa_handler = sigchld_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART;
    if (sigaction(SIGCHLD, &sa, NULL) == -1) {
        perror("sigaction");
        exit(EXIT_FAILURE);
    }

    while (1) {
        sin_size = sizeof(client_addr);
        new_fd = accept(sockfd, (struct sockaddr*)&client_addr, &sin_size);
        if (new_fd == -1) {
            perror("accept");
            continue;
        }
        printf("Received connection from %s\n", inet_ntoa(client_addr.sin_addr));

        if (!fork()) {
            close(sockfd);
            int num;
            ssize_t num_bytes;
            while ((num_bytes = recv(new_fd, &num, sizeof(num), 0)) > 0) {
                printf("Received number: %d\n", num);
            }
            close(new_fd);
            exit(EXIT_SUCCESS);
        }
        close(new_fd);
    }
    return 0;
}
