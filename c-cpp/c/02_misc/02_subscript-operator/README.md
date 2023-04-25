# Subscript operator

* The C standard defines the subscript operator `[]` as follows:
```
a[b] == *(a + b)
```
  * Therefore `a[5]` will evaluate to `*(a + 5)` and `5[a]` will evaluate
  to `*(5 + a)`, which are equivalent.

* In C++ we can go even further by overriding the `[]` operator.

## A weird construct

* Even if `a[5]` and `5[a]` evaluate to the same thing, it doesn't mean the
order are totally irrelevant. Let's consider the below code:

  ```C
  int* ptr = malloc(/*...*/);
  int* arr_plus_n(size_t n) {
      for (int i = 0; i < n; ++i) {
          ++ptr;
      }   
      return ptr;
  }
  int main(void) {
    for (int i = 0; i < ARRAY_SIZE; ++i) {
      ptr[i] = ARRAY_SIZE-i;
    }
    printf("%d\n", (arr_plus_n(1))[*arr_plus_n(2)]);
    return 0;
  }
  ```

  * What is the expected output?

* This code is cryptic because both the pointer and the index are function
calls. So the question boils down to: the pointer or the index, which
gets evaluated first?
  * To some's surprise, the order is unspecified in C and in C++ until C++17.

* But it doesn't matter that much, let's just this!

## References

* [C++17 creates a practical use of the backward array index operator](https://devblogs.microsoft.com/oldnewthing/20230403-00/?p=108005)
* [Order of evaluation of array indices (versus the expression) in C](https://stackoverflow.com/questions/59722807/order-of-evaluation-of-array-indices-versus-the-expression-in-c)