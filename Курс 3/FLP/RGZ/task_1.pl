task_1 :-
    write('Введите список : '),
    read(InputList),
    double_elements(InputList, DoubledList),
    write('Удвоенный список : '),
    format('Удвоенный список : ~w\n', [DoubledList]).

double_elements([], []).
double_elements([Head|Tail], [Head,Head|DoubledTail]) :-
    double_elements(Tail, DoubledTail).
