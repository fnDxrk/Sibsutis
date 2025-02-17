#include <arpa/inet.h>
#include <cstring>
#include <iostream>
#include <unistd.h>

int main()
{
    sockaddr_in server_addr {}, client_addr {};
    socklen_t client_len = sizeof(client_addr);
    char buffer[1024];

    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = 0;

    if (bind(sock, (sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("bind failed");
        close(sock);
        return 1;
    }

    socklen_t len = sizeof(server_addr);
    getsockname(sock, (sockaddr*)&server_addr, &len);
    std::cout << "Server listening on port " << ntohs(server_addr.sin_port) << "\n";

    while (true) {
        recvfrom(sock, buffer, 1024, 0, (sockaddr*)&client_addr, &client_len);
        int num = atoi(buffer);
        std::cout << "Received " << num << " from " << inet_ntoa(client_addr.sin_addr)
                  << ":" << ntohs(client_addr.sin_port) << "\n";
        num *= 2;
        sprintf(buffer, "%d", num);
        sendto(sock, buffer, strlen(buffer), 0, (sockaddr*)&client_addr, client_len);
    }
}
