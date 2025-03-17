#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/wait.h>

#define BUFFER_SIZE 1024
#define BACKLOG 10

const int enable = 1;

void handle_sigchld(int signal) {
    (void)signal;
    while (waitpid(-1, NULL, WNOHANG) > 0);
}

void handle_client(int client_sock) {
    char buffer[BUFFER_SIZE];
    while (1) {
        memset(buffer, 0, BUFFER_SIZE);
        ssize_t bytes_read = read(client_sock, buffer, BUFFER_SIZE - 1);
        if (bytes_read <= 0) {
            printf("Клиент отключился\n");
            break;
        }
        buffer[bytes_read] = '\0';
        printf("Получено: %s\n", buffer);
    }
    close(client_sock);
    exit(EXIT_SUCCESS);
}


int main() {
    signal(SIGCHLD, handle_sigchld);

    int server_sock, client_sock;
    struct sockaddr_in server_addr, client_addr;
    socklen_t addr_len = sizeof(server_addr);
    char buffer[BUFFER_SIZE];

    if ((server_sock = socket(AF_INET, SOCK_STREAM, 0)) == -1) {
        perror("socket");
        exit(EXIT_FAILURE);
    }

    setsockopt(server_sock, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(enable));

    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(0);
    server_addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

    if (bind(server_sock, (struct sockaddr *)&server_addr, sizeof(server_addr)) == -1) {
        perror("bind");
        close(server_sock);
        exit(EXIT_FAILURE);
    }

    if (getsockname(server_sock, (struct sockaddr *)&server_addr, &addr_len) == -1) {
        perror("getsockname");
        close(server_sock);
        exit(EXIT_FAILURE);
    }

    printf("Сервер запущен на %s:%d...\n", inet_ntoa(server_addr.sin_addr), ntohs(server_addr.sin_port));

    if (listen(server_sock, BACKLOG) < 0) {
        perror("listen");
        close(server_sock);
        exit(EXIT_FAILURE);
    }

    while (1) {
        if ((client_sock = accept(server_sock, (struct sockaddr *)&client_addr, &addr_len)) == -1) {
            perror("accept");
            continue;
        }

        pid_t pid = fork();
        if (pid == -1) {            // Ошибка создания процесса
            perror("fork");
            close(client_sock);
            continue;
        } else if (pid == 0) {      // Дочерний процесс
            close(server_sock);
            handle_client(client_sock);
        } else {                    // Родительский процесс
            close(client_sock);
        }
    }

    close(server_sock);

    return 0;
}