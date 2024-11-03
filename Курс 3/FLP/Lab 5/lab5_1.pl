% Предикат для печати всех нечётных чисел в порядке убывания
print_odd_descending(Current, Lower) :-

    % Логическое условие: если текущий элемент меньше нижней границы, остановиться
    Current < Lower, !.
    
print_odd_descending(Current, Lower) :-

    % Проверка на нечётность текущего числа
    Current >= Lower,
    Current mod 2 =:= 1,
    writeln(Current),
    Next is Current - 1,

    print_odd_descending(Next, Lower).

print_odd_descending(Current, Lower) :-

    % Если число чётное, просто идём дальше
    Current >= Lower,
    Next is Current - 1,

    print_odd_descending(Next, Lower).

startFirst :-

    % Ввод нижней границы диапазона
    writeln('Введите нижнюю границу диапазона:'),
    read(Lower),
    
    % Ввод верхней границы диапазона
    writeln('Введите верхнюю границу диапазона:'),
    read(Upper),
    
    % Проверка корректности диапазона
    (Upper >= Lower -> print_odd_descending(Upper, Lower)
        
    ;   writeln('Ошибка: верхняя граница должна быть больше или равна нижней.')
    ).