#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <termios.h>
#include <string.h>
#include <locale.h>
#include <sys/ioctl.h>

struct termios orig_term;
int width = 40, height = 20;

void restore_terminal(void) {
    tcsetattr(STDIN_FILENO, TCSANOW, &orig_term);
    printf("\033[0m\033[?25h");
    fflush(stdout);
}

void handle_sigint(int signum) {
    (void)signum;
    restore_terminal();
    exit(0);
}

int main(int argc, char *argv[]) {
    setlocale(LC_ALL, "C");
    if (argc < 6) {
        fprintf(stderr, "Usage: %s char dx dy color speed\n", argv[0]);
        fprintf(stderr, "Example: %s O 1 1 31 100000\n", argv[0]);
        return 1;
    }

    char ch = argv[1][0];
    int dx = atoi(argv[2]);
    int dy = atoi(argv[3]);
    int color = atoi(argv[4]);
    int delay = atoi(argv[5]);

    struct winsize w;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
    width = w.ws_col;
    height = w.ws_row;

    tcgetattr(STDIN_FILENO, &orig_term);
    atexit(restore_terminal);
    signal(SIGINT, handle_sigint);

    struct termios new_term = orig_term;
    new_term.c_lflag &= ~(ICANON | ECHO);
    new_term.c_cc[VMIN] = 1;
    new_term.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSANOW, &new_term);

    printf("\033[?25l");

    int x = width / 2;
    int y = height / 2;

    while (1) {
        printf("\033[%d;%dH\033[%dm%c", y, x, color, ch);
        fflush(stdout);
        usleep(delay);
        printf("\033[%d;%dH ", y, x);

        x += dx;
        y += dy;

        if (x <= 1 || x >= width) dx = -dx;
        if (y <= 1 || y >= height) dy = -dy;
    }

    restore_terminal();
    return 0;
}

