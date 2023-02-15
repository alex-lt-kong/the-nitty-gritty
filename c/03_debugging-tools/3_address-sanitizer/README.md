# AddressSanitizer

* [RTFM](https://github.com/google/sanitizers/wiki/AddressSanitizer)

* To make it work with cmake:
```
add_compile_options(-fsanitize=address -fno-omit-frame-pointer -g)
add_link_options(-fsanitize=address)
```