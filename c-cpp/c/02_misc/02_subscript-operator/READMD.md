# Subscript operator

* The C standard defines the subscript operator `[]` as follows:
```
a[b] == *(a + b)
```
    * Therefore `a[5]` will evaluate to `*(a + 5)` and `5[a]` will evaluate
    to `*(5 + a)`, which are equivalent.

* In C++ we can go even further by overriding the `[]` operator.
