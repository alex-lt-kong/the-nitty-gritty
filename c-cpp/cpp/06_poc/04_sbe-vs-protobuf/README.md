# Simple Binary Encoding vs Protocol Buffers

## SBE

* While SBE's main implementation is in Java, it does not mean that all
  other language versions are just bindings of its Java binary. It actually
  means that the main code generator is developed in Java.

* With the above in mind, even if we would like to generate code in C++, we
still need to build and run SBE's Java build: `./gradlew`

* Command to generate C++ code: `java -jar -Dsbe.generate.ir=true -Dsbe.target.language=Cpp -Dsbe.target.namespace=sbe -Dsbe.output.dir=/tmp/ ./simple-binary-encoding/sbe-all/build/libs/sbe-all-1.27.0.jar ~/repos/the-nitty-gritty/c-cpp/cpp/06_poc/04_sbe-vs-protobuf/sbe/tradeData.xml`

* This working PoC is roughly a combination of [this link](https://github.com/real-logic/simple-binary-encoding/blob/master/sbe-samples/src/main/cpp/GeneratedStubExample.cpp)
and [this link](https://www.baeldung.com/java-sbe).

## ProtoBuf

...

## Results

* SBE does appears to be an order of magnitude faster~

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

## Reference

* [Guide to Simple Binary Encoding](https://www.baeldung.com/java-sbe)
* [Simple Binary Encoding, a new ultra-fast marshalling API in C++, Java and .NET](https://weareadaptive.com/2013/12/10/sbe-1/)