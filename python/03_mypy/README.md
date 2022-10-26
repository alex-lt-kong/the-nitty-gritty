# Extra Proposal--enforce strong type in Python 
## 1. Basic idea
  * explicit is better than implicit.
  * Make Python strongly-typed via [mypy](https://github.com/python/mypy)--a linting tool that sponsored by Python

## 2. Mandatory use of type hinting for function parameters and return value
  * Without enforcing static type hinting, parameters decay to a "safe" equivalent of C's untyped pointer and
    functions are free to cast the pointer into any data structure, which causes unexpected behaviors here and there.
  * A [simple-example.py](./simple-example.py) on how `mypy` works.
  * `mypy` works with more advanced cases such as [function-pointers.py](./function-pointer.py)
  * Python supports an `object` type (or its even more general version, `Any`) which can effectively sideline
    `mypy`'s check if absolutely necessary. One example is [json-object.py](./json-object.py).
  * In a word, static typing won't make Python less flexible, Python still does what it can--just it forces programmers
    to think carefully on what is the expected behavior of a function.
  * **Proposal**: for new code, each commit must clear all the errors reported by `mypy` before being merged to master.
    For legacy code, I will try my best to make them compliant, but I need to take a closer look to be sure.
    
## 3. Discourage the use optional type hinting
  * Sometimes people define `get_price(as_of_date: [dt.date|dt.datetime|string|int])`.
  * It makes the function more diffucult to maintain, especially if the maintainer is another person--
    since it creates much more possible routes.
  * The proposed way is `get_price(as_of_date: dt.date)`: a date is a date, it can't be a "yyyy-mm-dd" string
    or yyyymmdd as an integer.
  * I would prefer enforcing this as an absolute ban at least for new code, if we change the wording to "try to avoid",
    people may abuse this exception--for existing code, I will try to implement it in this way, but it may not be
    easy if some optionality is deep-rooted.

3. Explicity define if a variable can be `None` by using `Union[str|None]`
    * this will be enforced by `mypy` automatically

4. The `Quality Assurance` should be in the README.md file of the repo, so that all maintainers will be able to refer to
   it easily.
