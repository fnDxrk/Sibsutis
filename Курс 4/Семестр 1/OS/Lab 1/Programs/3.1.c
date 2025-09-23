#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

int main() {
    struct winsize w;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);

    const char *message = "HELLO";
    int row = w.ws_row / 2;
    int col = (w.ws_col - strlen(message)) / 2;

    system("clear");

    for (int i = 0; i < row; i++) {
        printf("\n");
    }

    for (int i = 0; i < col; i++) {
        printf(" ");
    }

    printf("%s\n", message);

    return 0;
}
