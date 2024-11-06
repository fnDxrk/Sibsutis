print_odd_descending(Current, Lower) :-
    Current < Lower, !.
    
print_odd_descending(Current, Lower) :-
    Current >= Lower,
    (Current mod 2 =:= 1 -> writeln(Current) ; true),
    Next is Current - 1,
    print_odd_descending(Next, Lower).

startFirst :-

    writeln('Введите нижнюю границу диапазона:'),
    read(Lower),
    
    writeln('Введите верхнюю границу диапазона:'),
    read(Upper),
    
    (Upper >= Lower -> print_odd_descending(Upper, Lower)
        
    ;   writeln('Ошибка: верхняя граница должна быть больше или равна нижней.')
    ).
