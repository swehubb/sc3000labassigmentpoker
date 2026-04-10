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

% new succession rule (pure age)
new_succession_precedes(Person1, Person2) :- 
    older(Person1, Person2).
