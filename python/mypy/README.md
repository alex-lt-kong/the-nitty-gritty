0. The purpose is to make the code more explicit to other people and 
  the approach is to add static typing to Python via linting tools `mypy`

1. Mandatory explicit use of type hinting
    * Without enforcing type hinting, all Python variables decay to a "safe" equivalent of C's void pointer and
      functions are free to cast the pointer into any data structure, which causes unexpected behaviors here and there;
    * Each commit must clear all the errors reported by `mypy` before being merged to master;
    
2. Avoid optional type hinting
    * Sometimes people define `get_price(as_of_date: [dt.date|dt.datetime|string|int])`;
    * It makes the function more diffucult to maintain, especially if the maintainer is another person--
      since it creates much more possible routes;
    * The proposed way is `get_price(as_of_date: dt.date)`: a date is a date, it can't be a "yyyy-mm-dd" string
      or yyyymmdd as an integer.
    * I would prefer enforcing this as an absolute ban at least for new code, if we change the wording to "try to avoid",
      people may abuse this exception--for existing code, I will try to implement it in this way, but it may not be
      easy if some optionality is deep-rooted.

3. Explicity define if a variable can be `None` by using `Union[str|None]`
    * this will be enforced by `mypy` automatically

4. The `Quality Assurance` should be in the README.md file of the repo, so that all maintainers will be able to refer to
   it easily.
