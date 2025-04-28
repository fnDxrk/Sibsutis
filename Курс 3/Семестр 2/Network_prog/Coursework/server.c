#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#define PORT 8080
#define BUFFER_SIZE 1024
#define BACKLOG 10
#define MAX_HEADER_SIZE 1024

#define HTTP_200_OK "HTTP/1.1 200 OK\r\n"
#define HTTP_206_PARTIAL "HTTP/1.1 206 Partial Content\r\n"
#define HTTP_404_NOT_FOUND "HTTP/1.1 404 Not Found\r\n"
#define CONTENT_TYPE_HEADER "Content-Type: %s\r\n"
#define CONTENT_LENGTH_HEADER "Content-Length: %ld\r\n"
#define CONTENT_RANGE_HEADER "Content-Range: bytes %ld-%ld/%ld\r\n"
#define HEADER_END "\r\n"

#define CONTENT_TYPE_TEXT "text/plain"
#define CONTENT_TYPE_HTML "text/html"
#define CONTENT_TYPE_JPEG "image/jpeg"
#define CONTENT_TYPE_PNG "image/png"
#define DEFAULT_CONTENT_TYPE "application/octet-stream"

#define ERROR_404_MESSAGE "404 Not Found"

typedef struct {
    const char* ext;
    const char* mime_type;
} MimeType;

static const MimeType mime_types[] = {
    { "html", CONTENT_TYPE_HTML },
    { "htm", CONTENT_TYPE_HTML },
    { "txt", CONTENT_TYPE_TEXT },
    { "jpg", CONTENT_TYPE_JPEG },
    { "jpeg", CONTENT_TYPE_JPEG },
    { "png", CONTENT_TYPE_PNG }
};
static const size_t mime_types_count = sizeof(mime_types) / sizeof(MimeType);

const char* get_mime_type(const char* file_ext)
{
    if (!file_ext)
        return DEFAULT_CONTENT_TYPE;
    for (size_t i = 0; i < mime_types_count; i++) {
        if (strcasecmp(mime_types[i].ext, file_ext) == 0) {
            return mime_types[i].mime_type;
        }
    }
    return DEFAULT_CONTENT_TYPE;
}

void send_response(int client_socket, const char* file_path, off_t range_start)
{
    char header[MAX_HEADER_SIZE];
    char buffer[BUFFER_SIZE];

    const char* file_ext = strrchr(file_path, '.');
    file_ext = file_ext ? file_ext + 1 : "";

    int file_fd = open(file_path, O_RDONLY);
    if (file_fd == -1) {
        snprintf(buffer, sizeof(buffer),
            HTTP_404_NOT_FOUND
                CONTENT_TYPE_HEADER
                    HEADER_END
                        ERROR_404_MESSAGE,
            CONTENT_TYPE_TEXT);
        send(client_socket, buffer, strlen(buffer), 0);
        return;
    }

    struct stat st;
    if (fstat(file_fd, &st) < 0) {
        close(file_fd);
        return;
    }
    off_t file_size = st.st_size;
    const char* mime_type = get_mime_type(file_ext);

    if (range_start > 0 && range_start < file_size) {
        lseek(file_fd, range_start, SEEK_SET);
        snprintf(header, sizeof(header),
            HTTP_206_PARTIAL
                CONTENT_TYPE_HEADER
                    CONTENT_RANGE_HEADER
                        CONTENT_LENGTH_HEADER
                            HEADER_END,
            mime_type, range_start, file_size - 1, file_size, file_size - range_start);
    } else {
        snprintf(header, sizeof(header),
            HTTP_200_OK
                CONTENT_TYPE_HEADER
                    CONTENT_LENGTH_HEADER
                        HEADER_END,
            mime_type, file_size);
    }

    if (send(client_socket, header, strlen(header), 0) < 0) {
        close(file_fd);
        return;
    }

    ssize_t bytes_read;
    while ((bytes_read = read(file_fd, buffer, sizeof(buffer))) > 0) {
        if (send(client_socket, buffer, bytes_read, 0) < 0) {
            break;
        }
    }

    close(file_fd);
}

void* handle_client(void* arg)
{
    int client_socket = *((int*)arg);
    free(arg);

    char buffer[BUFFER_SIZE];
    ssize_t bytes_received = recv(client_socket, buffer, sizeof(buffer) - 1, 0);
    if (bytes_received <= 0) {
        close(client_socket);
        return NULL;
    }
    buffer[bytes_received] = '\0';

    off_t range_start = 0;
    char* range_header = strstr(buffer, "Range: bytes=");
    if (range_header) {
        range_header += strlen("Range: bytes=");
        range_start = atoll(range_header);
    }

    if (strncmp(buffer, "GET /", 5) == 0) {
        char* path_start = buffer + 5;
        char* path_end = strchr(path_start, ' ');
        if (path_end) {
            *path_end = '\0';
            if (strcmp(path_start, "") == 0) {
                path_start = "index.html";
            }
            send_response(client_socket, path_start, range_start);
        }
    }

    close(client_socket);
    return NULL;
}

int main()
{
    int server_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket < 0) {
        perror("socket");
        return EXIT_FAILURE;
    }

    int opt = 1;
    if (setsockopt(server_socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt))) {
        perror("setsockopt");
        close(server_socket);
        return EXIT_FAILURE;
    }

    struct sockaddr_in server_addr = {
        .sin_family = AF_INET,
        .sin_port = htons(PORT),
        .sin_addr.s_addr = INADDR_ANY
    };

    if (bind(server_socket, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("bind");
        close(server_socket);
        return EXIT_FAILURE;
    }

    if (listen(server_socket, BACKLOG) < 0) {
        perror("listen");
        close(server_socket);
        return EXIT_FAILURE;
    }

    printf("Server started on port %d\n", PORT);

    while (1) {
        int* client_socket = malloc(sizeof(int));
        if (!client_socket) {
            perror("malloc");
            continue;
        }

        *client_socket = accept(server_socket, NULL, NULL);
        if (*client_socket < 0) {
            perror("accept");
            free(client_socket);
            continue;
        }

        pthread_t thread_id;
        if (pthread_create(&thread_id, NULL, handle_client, client_socket)) {
            perror("pthread_create");
            close(*client_socket);
            free(client_socket);
        } else {
            pthread_detach(thread_id);
        }
    }

    close(server_socket);
    return EXIT_SUCCESS;
}
