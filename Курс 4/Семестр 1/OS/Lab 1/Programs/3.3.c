#include <stdio.h>
#include <unistd.h>
#include <termios.h>

int main(void) {
    int x, y;
    int dx = 1, dy = 1;
    int width = 40, height = 10;
    char ch = 'O';

    printf("\033[?25l");

    x = 1;
    y = 1;

    while (1) {
        printf("\033[%d;%dH%c", y, x, ch);
        fflush(stdout);
        usleep(100000);

        printf("\033[%d;%dH ", y, x);

        x += dx;
        y += dy;

        if (x <= 1 || x >= width) dx = -dx;
        if (y <= 1 || y >= height) dy = -dy;
    }

    printf("\033[?25h");
    return 0;
}
