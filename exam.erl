% This example is about parsing CSV to JSON using erl spawn,
% the main use case of this is when you parse giant CSV files (Big Data)
% and you would like to ingest the data in chunks.
% The whole parsing is in form of process chain and clien server pattern.

% To execute this file you need example csv file, csv_parser.erl
% To run the code simple run: erlc ./(exam.erl | csv_parser.erl) -> erl -> exam:main().
-module(exam).
-import(string, [len/1, concat/2, chr/2, substr/3, str/2,
				 to_lower/1, to_upper/1]).
-import(csv_parser, [usage/0, parse_file/1]).
-compile(export_all).

main() ->
    io:format("~p \n", [self()]),
    Csv_handler_process = spawn(exam, csv_handler, [self()]),
    
    register(csv_handler, Csv_handler_process),
    Reference = erlang:monitor(process, Csv_handler_process),
    Monitor_process = spawn(exam, monitor_main_process, [Reference, Csv_handler_process]),

    % Send read csv file
    CSV = "addresses.csv",
    Csv_handler_process ! [CSV],
    % Wait for message back with csv result
    % Wait for message back with json content

    % Send message to save the content as json file
    % Wait for message back with success flag
    loop_csv_to_json_process(self()).

monitor_main_process(Reference, Pid) ->
    receive
        {'DOWN', Reference, process, Pid, Reason} ->
            io:format("I (parent) My worker ~p died (~p)~n", [Pid, Reason]),
            main()
    end.

loop_csv_to_json_process(Pid) ->
    io:format("loop_csv_to_json_process called...\n", []),
    receive
        {Rows} ->
            io:format("Root Process got the rows \n", []),
            To_json_process = spawn(exam, to_json, [Pid, Rows]),
            % Send message to be parsed to json
            To_json_process ! "Any",
            loop_csv_to_json_process(Pid);
        {Json, _} ->
            io:format("CSV to JSON ready! \n", []),
            Write_to_json_file = spawn(exam, write_json_to_file, [Pid, Json]),
            Write_to_json_file ! "Any",
            loop_csv_to_json_process(Pid);
        {Status, _, _} ->
            io:format("JSON stored! \n", []);
        Any ->
            io:format("Default main thread handler... ~p", [Any]),
            loop_csv_to_json_process(Pid)
    end.

csv_handler(Pid) ->
    receive
        [CSV] ->
            io:format("Reading CSV... ~p \n", [CSV]),
            Read_CSV_Process = spawn(exam, read_csv_file, [self(), CSV]),
            Read_CSV_Process ! "Any",
            csv_handler(Pid);
        {Rows} ->
            io:format("Rows \n", []),
            Pid ! {Rows},
            csv_handler(Pid);
        Any ->
            io:format("Works or Rows \n", []),
            csv_handler(Pid)
    end.

read_csv_file(Pid, CSV) ->
    receive
        Any ->
            io:format("read_csv_file called \n", []),
            % Recive message to read csv file
            Rows = csv_parser:parse_file(CSV),
            % csv_parser:print_rows(Rows),
            % Send message with the content result to to_json
            Rows,
            io:format("Rows: ~p \n", [Rows]),
            Pid ! {Rows}
    end.

to_json(Pid, Rows) ->
    receive
        Any ->
            io:format("to_json called with... ~p ~p\n", [Any, Rows]),
            % Rows = read_csv_file(),
            Col_names = lists:nth(1, Rows),

            Col_names_list = tuple_to_list(Col_names),
            Json_map = maps:from_keys(Col_names_list, ok),
            
            for(length(Rows), 2, fun(Index) ->
                Row = lists:nth(Index, Rows),
                Row_list = tuple_to_list(Row),
                Json = maps:map(fun(Key, _) ->
                    Map_keys = lists:enumerate(Col_names_list),
                    Key_index_raw = lists:filter(fun(X) ->
                        {_,K} = X,
                        K == Key end, Map_keys),
                    
                    {Key_index,_} = lists:nth(1, Key_index_raw),
                    lists:nth(Key_index, Row_list) end, Json_map),
                    
                    % Send message to save the json file
                    io:fwrite("Json : ~p\n\n", [Json]),
                    Pid ! {Json, "just a flag"} end)
    end.

write_json_to_file(Pid, Json) ->
    % Receive message to save json file
    receive
        Any ->
            io:format("~p\n", [Json]),
            % The actual map to json code is out of the scope of this program
            % That's why at the end file only the values of the map are populated
            Status = file:write_file(lists:nth(1, maps:values(Json)) ++ ".json", [maps:values(Json)]),
            Pid ! {Status, "flag", "flag"}
    end.
    
for(0,_,_) ->
    ok;

for(Max,Min,Fn) when Max > 0 ->
    if
        Max < Min ->
            ok;
        true ->
            Fn(Max),
            for(Max-1,Min,Fn)
    end.