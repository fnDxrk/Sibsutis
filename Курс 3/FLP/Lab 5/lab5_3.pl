split_list([], _, _, [], [], []).  

split_list([Head|Tail], Min, Max, [Head|LessThanMin], Between, GreaterThanMax) :-
    Head < Min,
    split_list(Tail, Min, Max, LessThanMin, Between, GreaterThanMax).

split_list([Head|Tail], Min, Max, LessThanMin, [Head|Between], GreaterThanMax) :-
    Head >= Min,
    Head =< Max,
    split_list(Tail, Min, Max, LessThanMin, Between, GreaterThanMax).

split_list([Head|Tail], Min, Max, LessThanMin, Between, [Head|GreaterThanMax]) :-
    Head > Max,
    split_list(Tail, Min, Max, LessThanMin, Between, GreaterThanMax).

start :-
    writeln('Введите список (например, [1,2,3]):'),
    read(List),
    
    writeln('Введите первое число (меньший порог):'),
    read(Min),
    
    writeln('Введите второе число (больший порог):'),
    read(Max),
    
    (Min =< Max ->
        split_list(List, Min, Max, LessThanMin, Between, GreaterThanMax),
        format('Меньше ~w: ~w~n', [Min, LessThanMin]),
        format('От ~w до ~w: ~w~n', [Min, Max, Between]),
        format('Больше ~w: ~w~n', [Max, GreaterThanMax])
    ;
        writeln('Ошибка: первое число должно быть меньше или равно второму.')
    ).
