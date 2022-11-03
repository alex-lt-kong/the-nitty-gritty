# Cache associativity

* The general idea of cache associativity is that, say CPU wants to load a cache line from main memory from
cache memory, the cache line can only be stored in a few places of the cache, depending on its physicial address. 
A few informative videos explain this in a comprehensive manner:
    * [Introduction to Cache Memory](https://www.youtube.com/watch?v=Ez_kyBS-y5w)
    * [Direct Memory Mapping](https://www.youtube.com/watch?v=V_QS1HzJ8Bc)
    * [Associative Mapping](https://www.youtube.com/watch?v=uwnsMaH-iV0)
    * [Set Associative Mapping](https://www.youtube.com/watch?v=KhAh6thw_TI)
    * [Cache Memory Mapping - A Comparative Study](https://www.youtube.com/watch?v=e8RCnG2ibJk)

* The expected result should be something like this shown by Timur Doumler:
    <img src="./assets/expected-results.png" />


* CPU manufacturers like Intel may not document technical details such as the
nnumber of sets its CPUs' cache is using.

## References
* https://www.uops.info/cache.html#HSW
* https://en.wikichip.org/wiki/intel/microarchitectures/ice_lake_(client)
* https://en.wikichip.org/wiki/intel/microarchitectures/sandy_bridge_(client)