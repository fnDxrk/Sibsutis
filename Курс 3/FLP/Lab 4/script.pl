% 1 (родитель)
parent(john, bob).
parent(mary, bob).
parent(mary, ann).
parent(bob, liz).
parent(bob, paul).
parent(bob, sam).
parent(paul, pat).

% 2 (мужчина и женщина)
male(john).
male(bob).
male(pat).
male(paul).
male(sam).

female(mary).
female(ann).
female(liz).

% 3 (определение отношений)

father(X, Y) :- parent(X, Y), male(X).

mother(X, Y) :- parent(X, Y), female(X).

brother(X, Y) :- parent(Z, X), parent(Z, Y), male(X), X \= Y.

sister(X, Y) :- parent(Z, X), parent(Z, Y), female(X), X \= Y.

grandson(X, Y) :- parent(Y, Z), parent(Z, X), male(X).

aunt(X, Y) :- parent(Z, Y), sister(X, Z).

two_children(X) :- parent(X, A), parent(X, B), A \= B.

man_with_son(X) :- father(X, Y), male(Y).

% Queries (вопросы)
% father(Who, sam).
% mother(_, bob).
% sister(Who, sam).
% sister(_, liz).
% brother(Who, bob).
% grandson(Who, mary).
% grandson(paul, Who).
% aunt(Who, sam).
% brother(Brother, ann), father(Brother, Nephew), male(Nephew).
% sister(Sister, ann), father(Sister, Nephew), male(Nephew).
% two_children(Who).
% man_with_son(bob).
