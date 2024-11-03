% Предикат для вычисления числа Фибоначчи
fib(0, 1).  % F(0) = 1
fib(1, 1).  % F(1) = 1
fib(N, F) :- 
    N > 1,
    N1 is N - 1,
    N2 is N - 2,
    fib(N1, F1),
    fib(N2, F2),
    F is F1 + F2.

% Основной предикат для циклического ввода с помощью repeat
findFibonacci :-
    repeat,  % Начало цикла
    writeln('Введите номер числа Фибоначчи (отрицательное число для выхода): '),
    read(N),  % Считывание введённого номера (индекса)
    
    ( N < 0 ->  writeln('Выход из программы.'), !
    
    ;   fib(N, F),  % Вычисление числа Фибоначчи
        format('Число Фибоначчи с номером ~w: ~w~n', [N, F]),
        fail  % Возврат к началу repeat для нового ввода
    ).