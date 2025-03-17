#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <arpa/inet.h>
#include <sys/socket.h>

#define BUFFER_SIZE 1024
#define BACKLOG 10
#define FILE_NAME "server_data.txt"

pthread_mutex_t file_mutex = PTHREAD_MUTEX_INITIALIZER;
FILE* file;

const int enable = 1;

void* handle_client(void *arg) {
    int client_sock = *(int*)arg;
    free(arg);
    char buffer[BUFFER_SIZE];
    ssize_t bytes_read;

    while ((bytes_read = recv(client_sock, buffer, BUFFER_SIZE - 1, 0)) > 0) {
        buffer[bytes_read] = '\0';
        printf("Получено: %s\n", buffer);
        pthread_mutex_lock(&file_mutex);
        fprintf(file, "%s\n", buffer);
        fflush(file);
        pthread_mutex_unlock(&file_mutex);
    }
    close(client_sock);
    pthread_exit(NULL);
    return NULL;
}

int main() {
    int server_sock, client_sock;
    struct sockaddr_in server_addr, client_addr;
    socklen_t addr_size = sizeof(struct sockaddr_in);
    pthread_t thread_id;

    if ((server_sock = socket(AF_INET, SOCK_STREAM, 0)) == -1) {
        perror("socket");
        exit(EXIT_FAILURE);
    }
    
    setsockopt(server_sock, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(enable));

    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(0);
    server_addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

    if (bind(server_sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("bind");
        exit(EXIT_FAILURE);
    }

    if (getsockname(server_sock, (struct sockaddr *)&server_addr, &addr_size) == -1) {
        perror("getsockname");
        close(server_sock);
        exit(EXIT_FAILURE);
    }
    
    printf("Сервер запущен на %s:%d...\n", inet_ntoa(server_addr.sin_addr), ntohs(server_addr.sin_port));

    if (listen(server_sock, BACKLOG) < 0) {
        perror("listen");
        exit(EXIT_FAILURE);
    }

    file = fopen(FILE_NAME, "a");
    if (!file) {
        perror("file");
        exit(EXIT_FAILURE);
    }

    while(1) {
        int *client_sock_ptr = malloc(sizeof(int));
        if (!client_sock_ptr) {
            perror("malloc");
            continue;
        }

        if ((client_sock = accept(server_sock, (struct sockaddr*)&client_addr, &addr_size)) < 0) {
            perror("Ошибка accept");
            free(client_sock_ptr);
            continue;
        }

        *client_sock_ptr = client_sock;

        if (pthread_create(&thread_id, NULL, handle_client, client_sock_ptr) != 0) {
            perror("pthread_create");
            free(client_sock_ptr);
            continue;
        }

        pthread_detach(thread_id);
    }

    fclose(file);
    close(server_sock);
    pthread_mutex_destroy(&file_mutex);

    return 0;
}