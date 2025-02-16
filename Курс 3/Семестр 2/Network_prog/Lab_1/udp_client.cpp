#include <arpa/inet.h>
#include <cstring>
#include <iostream>
#include <unistd.h>

int main(int argc, char* argv[])
{
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <server_ip> <port>\n";
        return 1;
    }

    sockaddr_in server_addr {};
    char buffer[1024];
    int sock = socket(AF_INET, SOCK_DGRAM, 0);

    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(atoi(argv[2]));
    inet_pton(AF_INET, argv[1], &server_addr.sin_addr);

    int i = atoi(argv[3]);

    for (int j = 0; j < i; ++j) {
        sprintf(buffer, "%d", 1);
        sendto(sock, buffer, strlen(buffer), 0, (sockaddr*)&server_addr, sizeof(server_addr));
        recvfrom(sock, buffer, 1024, 0, nullptr, nullptr);
        std::cout << "Received from server: " << buffer << "\n";
        sleep(i);
    }
    close(sock);
}
