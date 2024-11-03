% Предикат для разбиения списка
split_list([], _, _, [], [], []).  % Базовый случай: пустой список

% Если голова списка меньше первого числа
split_list([Head|Tail], Min, Max, [Head|LessThanMin], Between, GreaterThanMax) :-
    Head < Min,
    split_list(Tail, Min, Max, LessThanMin, Between, GreaterThanMax).

% Если голова списка в пределах двух чисел
split_list([Head|Tail], Min, Max, LessThanMin, [Head|Between], GreaterThanMax) :-
    Head >= Min,
    Head =< Max,
    split_list(Tail, Min, Max, LessThanMin, Between, GreaterThanMax).

% Если голова списка больше второго числа
split_list([Head|Tail], Min, Max, LessThanMin, Between, [Head|GreaterThanMax]) :-
    Head > Max,
    split_list(Tail, Min, Max, LessThanMin, Between, GreaterThanMax).

% Предикат для ввода и разбиения списка
start :-
    % Ввод списка
    writeln('Введите список (например, [1,2,3]):'),
    read(List),
    
    % Ввод двух чисел
    writeln('Введите первое число (меньший порог):'),
    read(Min),
    
    writeln('Введите второе число (больший порог):'),
    read(Max),
    
    % Проверка, что первое число меньше второго
    (Min =< Max ->
        split_list(List, Min, Max, LessThanMin, Between, GreaterThanMax),
        % Вывод результатов
        format('Меньше ~w: ~w~n', [Min, LessThanMin]),
        format('От ~w до ~w: ~w~n', [Min, Max, Between]),
        format('Больше ~w: ~w~n', [Max, GreaterThanMax])
    ;
        writeln('Ошибка: первое число должно быть меньше или равно второму.')
    ).