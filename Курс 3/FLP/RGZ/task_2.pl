:- use_module(library(lists)).
:- use_module(library(apply)).

process_text_file(InputFile, OutputFile) :-
    open(InputFile, read, InStream),
    open(OutputFile, write, OutStream),
    process_lines(InStream, OutStream),
    close(InStream),
    close(OutStream).

process_lines(InStream, OutStream) :-
    read_line_to_codes(InStream, Line),
    ( Line == end_of_file
    -> true
    ;  atom_codes(Atom, Line),
       split_string(Atom, " ", "", Words),
       format_line(Words, FormattedLine),
       format(OutStream, '~s~n', [FormattedLine]),
       process_lines(InStream, OutStream)
    ).

format_line(Words, FormattedLine) :-
    length(Words, NumWords),
    NumWords > 1,
    maplist(string_length, Words, WordLengths),
    sum_list(WordLengths, TotalWordLength),
    SpacesNeeded is 80 - TotalWordLength,
    NumGaps is NumWords - 1,
    ExtraSpaces is SpacesNeeded mod NumGaps,
    BaseSpaces is SpacesNeeded // NumGaps,
    distribute_spaces(Words, BaseSpaces, ExtraSpaces, FormattedLine).

distribute_spaces([Word], _, _, Word).
distribute_spaces([Word1, Word2 | Words], BaseSpaces, ExtraSpaces, FormattedLine) :-
    ( ExtraSpaces > 0
    -> Spaces is BaseSpaces + 1, NewExtraSpaces is ExtraSpaces - 1
    ;  Spaces is BaseSpaces, NewExtraSpaces is ExtraSpaces
    ),
    generate_spaces(Spaces, SpacesAtom),
    atom_concat(Word1, SpacesAtom, Temp),
    distribute_spaces([Word2 | Words], BaseSpaces, NewExtraSpaces, Rest),
    atom_concat(Temp, Rest, FormattedLine).

generate_spaces(0, "").
generate_spaces(N, Spaces) :-
    N > 0,
    N1 is N - 1,
    generate_spaces(N1, Spaces1),
    atom_concat(" ", Spaces1, Spaces).

main :-
    write('Input file name: '),
    read_line_to_codes(user_input, InputCodes),
    atom_codes(InputFile, InputCodes),
    write('Output file name: '),
    read_line_to_codes(user_input, OutputCodes),
    atom_codes(OutputFile, OutputCodes),
    process_text_file(InputFile, OutputFile).
