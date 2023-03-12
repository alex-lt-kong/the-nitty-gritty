# Simple Binary Encoding vs Protocol Buffers

## SBE

* C++ code generation: `java -jar -Dsbe.generate.ir=true -Dsbe.target.language=Cpp -Dsbe.target.namespace=sbe  -Dsbe.output.dir=/tmp/ sbe-all/build/libs/sbe-all-1.27.0.jar ~/repos/the-nitty-gritty/c-cpp/cpp/06_poc/04_sbe-vs-protobuf/sbe/tradeData.xml`

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