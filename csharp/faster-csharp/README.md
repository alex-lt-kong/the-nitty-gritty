# faster-csharp

Experiments that make C# faster with**OUT** optimizing the time complexity of algorithms.

# Results

===== GC.Collect() vs Loop =====\
50,000 iterations: 238 ms\
GC.Collect(): 2897 ms\
"a GC.Collect() call is worth 50 thousand iterations"


===== Static vs Dynamic Arraies =====\
vanillaArray: 90 ms\
preAllocatedArrayList: 2817 ms\
naiveArrayList: 2054 ms


===== Array vs ArrayPool =====\
Array: 153 ms\
ArrayPool: 12 ms


===== Class vs Struct vs Dict =====\
Class: 241 ms\
Struct: 87 ms\
Dictionary: 2402 ms


===== Try vs No try =====\
try{}ed: 84 ms\
not try{}ed: 66 ms\
"There IS harm in try{}ing"


===== Finalizer vs No Finalizer =====\
Class: 234 ms\
Class with a finalizer: 395 ms


===== String vs StringBuilder =====\
StringBuilder: 127 ms\
String: 38494 ms


===== Small Object Heap vs Large Object Heap ====
Small Object Heap: 1244 ms\
Big Object Heap: 3574 ms


===== Hashtable vs Dictionary =====\
Same-type hashtable: 11 ms\
Different-type hashtable: 1004 ms\
Dictionary: 5 ms

===== Division vs Reciprocal Multiplication =====\
Naive division: 201 ms\
Naive reciprocal multiplication: 233 ms\
Obfuscated division: 653 ms\
Obfuscated reciprocal multiplication: 663 ms

===== gRPC vs RESTful API =====\
gRPC: 11 ms\
RESTful API: 2082 ms
