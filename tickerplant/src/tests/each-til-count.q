countries:flip `code`name`population`ts!"s*ip"$\:();

`countries insert (`jp;name:enlist"Japan";100;.z.P)
`countries insert (`us;name:enlist"United States";300;.z.P)
`countries insert (`fr;name:enlist"France";50;.z.P)

show "show entire table:"
show countries

show "til count works together to generate a list of indexes"
show "(but no, each does NOT fit here):"
til count countries

show "we can print one row like this:"
countries 0
countries 1
countries 2

show_count_func:{show x}
show_row_func:{show countries x}

show "each only works with function:"
show_count_func each til count countries
show_row_func each til count countries

exit 0