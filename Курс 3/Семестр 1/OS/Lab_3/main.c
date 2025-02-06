#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

void
exp_2()
{
  pid_t pid = fork();

  if (pid < 0) {
    perror("fork failed");
    exit(EXIT_FAILURE);
  }

  if (pid == 0) {
    printf("This is the child process with PID: %d\n", getpid());
    printf("Parent PID: %d\n", getppid());
    sleep(10);
  } else {
    printf("This is the parent process with PID: %d\n", getpid());
    printf("Child PID: %d\n", pid);
    sleep(10);
  }
}

void
exp_3()
{
  pid_t pid1, pid2;

  pid1 = fork();
  if (pid1 < 0) {
    perror("fork failed");
    exit(EXIT_FAILURE);
  }

  if (pid1 == 0) {
    // Первый дочерний процесс
    // printf("First child process PID: %d\n", getpid());
    // printf("Parent PID: %d\n", getppid());
    sleep(5);
    exit(EXIT_SUCCESS);
  } else {
    // Родительский процесс продолжает выполнение
    pid2 = fork();
    if (pid2 < 0) {
      perror("fork failed");
      exit(EXIT_FAILURE);
    }

    if (pid2 == 0) {
      // Второй дочерний процесс
      // printf("Second child process PID: %d\n", getpid());
      // printf("Parent PID: %d\n", getppid());
      sleep(5);
      exit(EXIT_SUCCESS);
    } else {
      // Родительский процесс
      printf("Parent process PID: %d\n", getpid());
      printf("First child PID: %d\n", pid1);
      printf("Second child PID: %d\n", pid2);
      sleep(30);
    }
  }
}

int
main()
{
  // exp_2();
  exp_3();

  return 0;
}
