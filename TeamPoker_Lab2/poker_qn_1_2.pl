% facts
competitor(sumsum, appy).
developed(sumsum, galactica_s3).
stole(stevey, galactica_s3).
boss(stevey, appy).
smart_phone_technology(galactica_s3).

% rule
rival(X, Y) :-
    competitor(X, Y).
business(X) :-
    smart_phone_technology(X).

unethical(X) :-
    boss(X, Y),
    rival(W, Y),
    developed(W, Z),
    stole(X, Z),
    business(Z).
