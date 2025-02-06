most_frequent_elements :-
    write('Введите список: '),
    read(List),
    count_frequencies(List, [], FrequencyList),
    max_frequency(FrequencyList, MaxFreq),
    collect_most_frequent(FrequencyList, MaxFreq, MostFrequent),
    format('Наиболее часто встречающиеся элементы: ~w~n', [MostFrequent]).

count_frequencies([], Acc, Acc).
count_frequencies([H|T], Acc, FrequencyList) :-
    update_frequency(H, Acc, UpdatedAcc),
    count_frequencies(T, UpdatedAcc, FrequencyList).

update_frequency(Element, [], [(Element, 1)]).
update_frequency(Element, [(Element, Freq)|T], [(Element, NewFreq)|T]) :-
    NewFreq is Freq + 1.
update_frequency(Element, [(OtherElement, Freq)|T], [(OtherElement, Freq)|UpdatedT]) :-
    Element \= OtherElement,
    update_frequency(Element, T, UpdatedT).

max_frequency([( _, Freq)], Freq).
max_frequency([(_, Freq)|T], MaxFreq) :-
    max_frequency(T, TailMaxFreq),
    MaxFreq is max(Freq, TailMaxFreq).

collect_most_frequent([], _, []).
collect_most_frequent([(Element, Freq)|T], MaxFreq, [Element|MostFrequent]) :-
    Freq =:= MaxFreq,
    collect_most_frequent(T, MaxFreq, MostFrequent).
collect_most_frequent([(_, Freq)|T], MaxFreq, MostFrequent) :-
    Freq \= MaxFreq,
    collect_most_frequent(T, MaxFreq, MostFrequent).
