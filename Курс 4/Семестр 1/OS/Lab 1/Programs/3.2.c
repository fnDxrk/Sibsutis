#include <stdio.h>
#include <stdlib.h>
#include <termios.h>
#include <unistd.h>

struct termios orig_term;

void restore_terminal(void) {
    tcsetattr(STDIN_FILENO, TCSANOW, &orig_term);
    printf("\033[0m\033[?25h");
}

int main(void) {
    struct termios new_term;
    unsigned char c;

    if (tcgetattr(STDIN_FILENO, &orig_term) != 0) return 1;
    atexit(restore_terminal);

    new_term = orig_term;
    new_term.c_lflag &= ~(ICANON | ECHO);
    new_term.c_cc[VMIN] = 1;
    new_term.c_cc[VTIME] = 0;
    if (tcsetattr(STDIN_FILENO, TCSANOW, &new_term) != 0) return 1;

    printf("\033[2J\033[H");
    printf("Press keys (ESC to exit):\n");

    while (1) {
        ssize_t n = read(STDIN_FILENO, &c, 1);
        if (n <= 0) break;
        if (c == 27) break;
        printf("Key code: %d\n", c);
    }

    return 0;
}
