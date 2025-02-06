:- use_module(library(lists)).

task_2(InputFile, OutputFile) :-
    process_file(InputFile, OutputFile).

process_file(InputFile, OutputFile) :-
    open(InputFile, read, InStream),
    open(OutputFile, write, OutStream),
    process_lines(InStream, OutStream),
    close(InStream),
    close(OutStream).

process_lines(InStream, OutStream) :-
    read_line_to_string(InStream, CurrentLine),
    ( CurrentLine == end_of_file
    -> true
    ;  split_string(CurrentLine, " ", "", WordsList),
       justify_words(WordsList, JustifiedLine),
       writeln(OutStream, JustifiedLine),
       process_lines(InStream, OutStream)
    ).

justify_words(WordsList, JustifiedLine) :-
    length(WordsList, WordCount),
    (WordCount =< 1 -> atomic_list_concat(WordsList, " ", JustifiedLine)
    ;
    maplist(string_length, WordsList, WordLengths),
    sum_list(WordLengths, TotalWordLength),
    RequiredSpaces is 80 - TotalWordLength,
    NumberOfGaps is WordCount - 1,
    BaseSpaces is RequiredSpaces // NumberOfGaps,
    ExtraSpaces is RequiredSpaces mod NumberOfGaps),
    distribute_spaces(WordsList, BaseSpaces, ExtraSpaces, JustifiedLine).

distribute_spaces([Word], _, _, Word).     
distribute_spaces([FirstWord, SecondWord | RemainingWords], BaseSpaces, ExtraSpaces, JustifiedLine) :-
    ( ExtraSpaces > 0
    -> ExtraSpacesForThisGap is BaseSpaces + 1,
       NewExtraSpaces is ExtraSpaces - 1
    ;  ExtraSpacesForThisGap is BaseSpaces,
       NewExtraSpaces is ExtraSpaces
    ),
    create_spaces(ExtraSpacesForThisGap, SpaceString),
    string_concat(FirstWord, SpaceString, TempLine),
    distribute_spaces([SecondWord | RemainingWords], BaseSpaces, NewExtraSpaces, RestLine),
    string_concat(TempLine, RestLine, JustifiedLine).

create_spaces(0, "").
create_spaces(Number, Spaces) :-
    Number > 0,
    NextNumber is Number - 1,
    create_spaces(NextNumber, SubSpaces),
    string_concat(" ", SubSpaces, Spaces).

