/ reference: https://code.kx.com/q/database/object/

/countries:([] name:();code:`symbol$();population:`int$())
countries:flip `code`name`population`ts!"s*ip"$\:();
/ https://code.kx.com/q/basics/datatypes/
/ Just like C, KDB treats string as an array of characters;
/ According to this link, https://stackoverflow.com/questions/53640171/kdb-how-to-assign-string-datatype-to-all-columns :
/ KDB does not allow to define column types as list during creation of table.
/ So that means you can not define your column type as String because that is also a list.
/ To circumvent this issue, we define a column `name whose type is left empty
/ then we insert rows to make it a string column

`countries insert (`jp;name:enlist"Japan";100;.z.P)
`countries insert (`us;name:enlist"United States";300;.z.P)
`countries insert (`fr;name:enlist"France";50;.z.P)
show countries
save `countries

cities:([] code:`symbol$(); name:(); gdp:`int$())
/ In SQL, one can declare one or more column(s) of a table as a primary key.
/ This means that the values in the column(s) are unique over the domain of the 
/ rows, making it possible to identify and retrieve a row via its key value. 
/ These two features motivate how q implements a keyed table.
/ In q, key column(s) are placed between the square brackets
/ Note that a keyed table is not a table – it is a dictionary and so has type 99h.
`cities insert (`ny;name:enlist"New York";gdp:100)
`cities insert (`london;name:enlist"London";110)
`cities insert (`hk;name:enlist"Hong Kong";120)
show cities
save `cities

exit 0