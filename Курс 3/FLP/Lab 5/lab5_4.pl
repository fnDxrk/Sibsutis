% Основной предикат
most_frequent_elements :-
    % Запрашиваем у пользователя список
    write('Введите список: '),
    read(List),
    % Подсчитываем частоту каждого элемента
    count_frequencies(List, [], FrequencyList),
    % Определяем максимальную частоту
    max_frequency(FrequencyList, MaxFreq),
    % Находим элементы с максимальной частотой
    collect_most_frequent(FrequencyList, MaxFreq, MostFrequent),
    % Выводим результат
    format('Наиболее часто встречающиеся элементы: ~w~n', [MostFrequent]).

% Подсчет количества вхождений каждого элемента в список пар (элемент-частота)
count_frequencies([], Acc, Acc).
count_frequencies([H|T], Acc, FrequencyList) :-
    update_frequency(H, Acc, UpdatedAcc),
    count_frequencies(T, UpdatedAcc, FrequencyList).

% Обновление списка частот: либо добавляем новый элемент, либо увеличиваем частоту
update_frequency(Element, [], [(Element, 1)]).
update_frequency(Element, [(Element, Freq)|T], [(Element, NewFreq)|T]) :-
    NewFreq is Freq + 1.
update_frequency(Element, [(OtherElement, Freq)|T], [(OtherElement, Freq)|UpdatedT]) :-
    Element \= OtherElement,
    update_frequency(Element, T, UpdatedT).

% Нахождение максимальной частоты
max_frequency([( _, Freq)], Freq).
max_frequency([(_, Freq)|T], MaxFreq) :-
    max_frequency(T, TailMaxFreq),
    MaxFreq is max(Freq, TailMaxFreq).

% Сбор элементов с максимальной частотой
collect_most_frequent([], _, []).
collect_most_frequent([(Element, Freq)|T], MaxFreq, [Element|MostFrequent]) :-
    Freq =:= MaxFreq,
    collect_most_frequent(T, MaxFreq, MostFrequent).
collect_most_frequent([(_, Freq)|T], MaxFreq, MostFrequent) :-
    Freq \= MaxFreq,
    collect_most_frequent(T, MaxFreq, MostFrequent).