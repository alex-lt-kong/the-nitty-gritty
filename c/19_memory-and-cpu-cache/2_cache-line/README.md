# Examine the size of cache line

* Data are transferred between memory and cache in blocks of fixed size, called cache lines. Usually the size of
a cache line is 64 bytes.

* Running the code, the expected result is that the loop should take more or less the same amount of time to
complete before the step size reaches 16 (i.e. 64 bytes) and the time needed after that should be halved each
time step size is doubled. 

## Results

* My results

![](./assets/my-results.png)

* Results are roughly consistent with the expected--before 16, the amount of time needed is more or less stable and
after that the time needed is approximately halved as step size doubles.

* There is one significantly different pattern: when the step size is between 1 and 16, the amount of time needed
is higher as step size gets closers to 1 ot 16 and lower when step size is closer to 8.

* (Failed) attempts to eliminate this unexpected pattern:
  * Try different computers, both virtual and physical ones--so that virtual machine won't complicate the issue.
  * Iterate each step size a few times--so that some random peaks will be smoothed out.
  * Simplify the calculation within the main loop--so that calculations won't be the bottleneck.
  * Disable vectorization and print out sample data--so that compilers can't just optimize my loop away.

* Other people results found online

![](./assets/igoro-results.png) From [Igoro Ostrovsky](http://igoro.com/archive/gallery-of-processor-cache-effects/)

![](./assets/timur-results.png) From [Timur Doumler](https://isocpp.org/blog/2017/05/cppcon-2016-want-fast-cpp-know-your-hardware-timur-doumler)