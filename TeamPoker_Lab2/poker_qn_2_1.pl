% royal family genders
male(charles).
male(andrew).
male(edward).
female(ann).

% birth order (1 is oldest)
birth_order(charles, 1).
birth_order(ann, 2).
birth_order(andrew, 3).
birth_order(edward, 4).

% figure out who is older
older(Person1, Person2) :- 
    birth_order(Person1, Pos1), 
    birth_order(Person2, Pos2), 
    Pos1 < Pos2.

% old succession rules
% males jump ahead of females
succession_precedes(Person1, Person2) :- 
    male(Person1), 
    female(Person2).

% if both are guys, the older one goes first
succession_precedes(Person1, Person2) :- 
    male(Person1), 
    male(Person2), 
    older(Person1, Person2).

% if both are girls, the older one goes first
succession_precedes(Person1, Person2) :- 
    female(Person1), 
    female(Person2), 
    older(Person1, Person2).
