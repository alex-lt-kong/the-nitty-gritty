# Copy elision

* Copy elision is a "minor" but very significant optimization.

* Consider the following case (Let's ignore function inline, compile-time
computation, etc):
    ```C++
    vector<double> GetScores() {
        vector<double> scores{ 1.414, 3.141, 2.71, 99, -1, 0.001 };
        return scores;
    }
    int main() {
        vector<double> myScores = GetScores();
        return 0;
    }    
    ```
    * What happens when we `return scores;`? By rights: `scores` is about to
    be out of scope and destroyed; a copy of `scores` is prepared and 
    assigned to `myScores`.

* This is worrying--what if `scores` is a vector with 1 million elements? We
are going to copy all of them?

* No, this can't happen and it is where copy elision kicks in.

* In general, the C++ standard allows a compiler to perform any optimization,
provided the resulting executable exhibits the same observable behaviour as
if (i.e. pretending) all the requirements of the standard have been fulfilled.
This is commonly referred to as the "as-if rule".
    * ISO C++ standard allows the omission of the copy and move construction
    of class objects, even if the copy/move constructor and the destructor
    have observable side-effects. This is an exception of the "as-if" rule
    and it is known as "copy elision".


## References

* [Wikipedia - Copy elision](https://en.wikipedia.org/wiki/Copy_elision)
* [Working Draft, Standard for Programming Language C++](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2020/n4849.pdf)
* [Cppconference.com - Copy elision](https://en.cppreference.com/w/cpp/language/copy_elision)