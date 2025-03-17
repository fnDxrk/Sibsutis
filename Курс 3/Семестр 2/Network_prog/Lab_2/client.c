#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>

#define BUFFER_SIZE 1024

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <IP> <port> <num_i>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    const char *server_ip = argv[1];
    int port = atoi(argv[2]);
    int i = atoi(argv[3]);

    if (i <= 0) {
        printf("i > 0!");
        exit(EXIT_FAILURE);
    }

    int sock;
    struct sockaddr_in server_addr;
    char buffer[BUFFER_SIZE];

    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) == -1) {
        perror("socket");
        exit(EXIT_FAILURE);
    }

    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);

    if (inet_pton(AF_INET, server_ip, &server_addr.sin_addr) <= 0) {
        perror("invalid IP");
        exit(EXIT_FAILURE);
    }

    if (connect(sock, (struct sockaddr *)&server_addr, sizeof(server_addr)) == -1) {
        perror("connect");
        exit(EXIT_FAILURE);
    }

    for (int j = 1; j <= i; j++) {
        snprintf(buffer, BUFFER_SIZE, "%d", j);
        if (write(sock, buffer, strlen(buffer)) == -1) {
            perror("send");
            exit(EXIT_FAILURE);
        }

        printf("Отправлено: %d\n", j);
        sleep(j);

        if (j < i) {
            sleep(j);
        }
    }

    close(sock);
    return 0;
}