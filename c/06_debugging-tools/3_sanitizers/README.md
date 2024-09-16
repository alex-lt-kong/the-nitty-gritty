# AddressSanitizer

* [RTFM](https://github.com/google/sanitizers/wiki/AddressSanitizer)

* To make it work with cmake:
```
add_compile_options(-fsanitize=address -fno-omit-frame-pointer -g)
add_link_options(-fsanitize=address)
```

# MemorySanitizer

* [RTFM](https://github.com/google/sanitizers/wiki/MemorySanitizer)

## Limitations

* MemorySanitizer is not a "plug-in"-style checker as shown in this
[false positive example](./msan/)

* Say we have a source code file `main.cpp` and we want to compile it to
binary file `main.out`, simply adding
`-fsanitize=memory -fsanitize-memory-track-origins` to my compilation
command is not enough. To avoid false positives, all the libraries used by
`main.out` must be compiled with
`-fsanitize=memory -fsanitize-memory-track-origins` as well.
  * this is called "instrumentation".

* This [official wiki post](https://github.com/google/sanitizers/wiki/MemorySanitizerLibcxxHowTo)
describes how we can achieve it.