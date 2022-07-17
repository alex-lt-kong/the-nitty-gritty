* Wasm: `emcc func.c -O3 -s WASM=1 -Wall -s MODULARIZE=1 -s BINARYEN_ASYNC_COMPILATION=0 -s EXPORTED_FUNCTIONS="['_malloc', '_free']" -o func.js`.
* C: `gcc main.c -o main.out -O3`.
* Execute: `node main.js`.
* Sample results
```
Results in milliseconds:
quickSort in JS:    2184,2134,2089,2082,2070,2048,2227,2204,2009,2056,2020,2141,2050,2150,2095,2261, avg: 2113
quickSort in Wasm:  186,183,181,192,186,181,183,183,183,182,184,183,183,181,178,183, avg: 183
Array.sort() in V8: 145,145,143,143,146,146,143,144,146,145,144,144,145,144,143,145, avg: 144 // Google's dark art ¯\_(ツ)_/¯ (explained: https://v8.dev/blog/array-sort)
quickSort in C:     176,165,161,162,162,163,163,161,163,164,162,195,164,165,162,176, avg: 166
```