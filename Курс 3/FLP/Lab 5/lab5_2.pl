fib(0, 1).
fib(1, 1).
fib(N, F) :- 
    N > 1,
    N1 is N - 1,
    N2 is N - 2,
    fib(N1, F1),
    fib(N2, F2),
    F is F1 + F2.

findFibonacci :-
    repeat,  
    writeln('Введите номер числа Фибоначчи (отрицательное число для выхода): '),
    read(N),  
    
    ( N < 0 ->  writeln('Выход из программы.'), !
    
    ;   fib(N, F),
        format('Число Фибоначчи с номером ~w: ~w~n', [N, F]),
        fail  
    ).
