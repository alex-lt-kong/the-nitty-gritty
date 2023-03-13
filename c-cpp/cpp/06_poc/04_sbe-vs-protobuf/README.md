# Simple Binary Encoding vs Protocol Buffers

## Results

* SBE does appear to be an order of magnitude faster。

* SBE result
```
Iterated 10 mil times
Elapsed 335.04ms (29.8472 mil records per sec or 33 ns per record)
```

* ProtoBuf result
```
Iterated 10 mil times
Elapsed 7982.7ms (1.25271 mil records per sec or 798 ns per record)
```

## Build and reproduce
### SBE

* While SBE's main implementation is in Java, it does not mean that all
  other language versions are just bindings of its Java binary. It actually
  means that the main code generator is developed in Java.

* With the above in mind, even if we would like to generate code in C++, we
still need to build and run SBE's Java build: `./gradlew`

* Command to generate C++ code: `java -jar -Dsbe.generate.ir=true -Dsbe.target.language=Cpp -Dsbe.target.namespace=sbe -Dsbe.output.dir=/tmp/ ./simple-binary-encoding/sbe-all/build/libs/sbe-all-1.27.0.jar ~/repos/the-nitty-gritty/c-cpp/cpp/06_poc/04_sbe-vs-protobuf/sbe/tradeData.xml`

* This working PoC is roughly a combination of [this link][3] and
[this link][2].

### ProtoBuf

...

## Important notes

* The power of SBE originates from its message structure. SBE is well suited
for fixed-size data like numbers, bitsets, enums, and arrays.

* A common use case for SBE is financial data streaming--mostly containing
numbers and enums--which SBE is specifically designed for. On the other hand,
SBE isn't well suited for variable-length data types like string and blob.
The reason for that is we most likely don't know the exact data size ahead.
Accordingly, this will end up with additional calculations at the streaming
time to detect the boundaries of data in a message.

* Although SBE still supports String and Blob data types, they're always
placed at the end of the message to keep the impact of variable length
calculations at a minimum.

## Reference

* [Guide to Simple Binary Encoding](https://www.baeldung.com/java-sbe)
* [Simple Binary Encoding, a new ultra-fast marshalling API in C++, Java and .NET](https://weareadaptive.com/2013/12/10/sbe-1/)

## References

1. [Guide to Simple Binary Encoding][1]
1. [Simple Binary Encoding, a new ultra-fast marshalling API in C++, Java and .NET][2]


[1]: https://www.baeldung.com/java-sbe "Guide to Simple Binary Encoding"
[2]: https://weareadaptive.com/2013/12/10/sbe-1/ "Simple Binary Encoding, a new ultra-fast marshalling API in C++, Java and .NET"
[3]: https://github.com/real-logic/simple-binary-encoding/blob/master/sbe-samples/src/main/cpp/GeneratedStubExample.cpp "SBE's official generated stub example"
[4]: https://a-teaminsight.com/blog/qa-fred-malabre-of-fpl-on-high-performance-trading-with-fix/ "Q&A: Fred Malabre of FPL on High Performance Trading with FIX"