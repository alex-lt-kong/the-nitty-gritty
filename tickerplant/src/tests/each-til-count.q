countries:flip `code`name`population`ts!"s*ip"$\:();

`countries insert (`jp;name:enlist"Japan";100;.z.P)
`countries insert (`us;name:enlist"United States";300;.z.P)
`countries insert (`fr;name:enlist"France";50;.z.P)

-1"show entire table:";
show countries

-1"\n\ntil count works together to generate a list of indexes";
show "(but no, each does NOT fit here):"
til count countries

-1"\n\nwe can print one row like this:";
countries 0
countries [1]
countries 2

show_count_func:{ show x; }
show_row_func:{show countries x}

-1"\n\neach only works with function:";
show_count_func each til count countries
-1"\n";
show_row_func each til count countries

exit 0
/ usage of -1 can be found:
/ https://code.kx.com/q/basics/handles/ and
/ https://community.kx.com/t5/New-kdb-q-users-question-forum/Show-or-0N/td-p/10750