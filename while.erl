% while program

-module(while).
-export([start/0]).

while(L) -> while(L,0).
while([], Acc) -> Acc;

while([_|T], Acc) ->
    io:fwrite("~w~n",[Acc]),
    io:fwrite("~w~n",[T]),
    while(T,Acc+1).

start() ->
    X = [1,2,3,4],
    while(X).