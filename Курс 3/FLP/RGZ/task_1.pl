double_elements([], []).
double_elements([H|T], [H,H|R]) :-
    double_elements(T, R).

main :-
    write('Input list: '),
    read(List),
    double_elements(List, DoubledList),
    write('Doubled list: '),
    write(DoubledList), nl.
