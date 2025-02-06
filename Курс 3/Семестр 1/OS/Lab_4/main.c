#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

int
main()
{
  pid_t children_pids[2];
  int i;

  // Запускаем 2 дочерних процесса
  for (i = 0; i < 2; ++i) {
    pid_t pid = fork();

    if (pid < 0) {
      fprintf(stderr, "Ошибка при вызове fork()\n");
      return 1;
    } else if (pid == 0) {
      // Дочерний процесс
      printf(
        "Дочерний процесс запущен. PID: %d, PPID: %d\n", getpid(), getppid());

      // Пример программы для запуска: ls
      char* args[] = { "ls", "-l", NULL };

      // Заменяем текущий процесс программой ls
      execvp(args[0], args);

      // Если execvp возвращает, значит произошла ошибка
      fprintf(stderr, "Ошибка при вызове execvp()\n");
      return 1;
    } else {
      // Родительский процесс
      children_pids[i] = pid;
      printf("Родительский процесс. PID: %d, запустил дочерний процесс с PID: "
             "%d\n",
             getpid(),
             pid);
    }
  }

  // Ожидаем завершения всех дочерних процессов
  for (i = 0; i < 2; ++i) {
    int status;
    waitpid(children_pids[i], &status, 0);
    printf("Дочерний процесс с PID %d завершился.\n", children_pids[i]);
  }

  return 0;
}
