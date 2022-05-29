% hello world program

-module(hello). 
-export([start/0]). 

start() -> 
   % M1 = #{name=>john,age=>25},
   % io:fwrite("~w",[M1]),

   % M2 = #{name=>doe,age=>24},
   % io:fwrite("~w",[M2]),

   % X = 40,
   % io:fwrite("~w",[X]),

   X = 40.00, 
   Y = 50.00, 
   io:fwrite("~f~n",[X]), 
   io:fwrite("~e",[Y]),
   
   io:fwrite("Hello, world!\n").