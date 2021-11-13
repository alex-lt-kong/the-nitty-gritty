countries:flip `code`name`population`ts!"s*ip"$\:();

`countries insert (`jp;name:enlist"Japan";100;.z.P)
`countries insert (`us;name:enlist"United States";300;.z.P)
`countries insert (`fr;name:enlist"France";50;.z.P)
show countries
countries_enum: .Q.en[`:splayed_tables] countries
`:splayed_tables/countries/ set countries_enum
/ Instead of calling set `countries, which is the intuitive equivalent of save `countries
/ here we need two lines: .Q.en makes a "sym" file containing the enumeration
/ of all symbols. We have to create an enumeration dict before we can save an
/ in-memory table as a spalyed table; otherwise a 'type error will be returned

cities:([] code:`symbol$(); name:(); gdp:`int$())
`cities insert (`ny;name:enlist"New York";gdp:100)
`cities insert (`london;name:enlist"London";110)
`cities insert (`hk;name:enlist"Hong Kong";120)
show cities
cities_enum: .Q.en[`:splayed_tables] cities
`:splayed_tables/cities/ set cities_enum
/ Note that if we enumerate two tables into the same "splayed_tables" dir
/ they share the same "sym" file. This can be confirmed by opening the
/ sym file with a hex editor.


exit 0