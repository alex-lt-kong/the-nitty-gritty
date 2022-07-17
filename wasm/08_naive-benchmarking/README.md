* Wasm: `emcc func.c -s WASM=1 -Wall -s MODULARIZE=1 -s BINARYEN_ASYNC_COMPILATION=0 -s EXPORTED_FUNCTIONS="['_malloc', '_free']" -o func.js`.
* C: `gcc main.c -o main.out -O3`.
* Execute: `node main.js`.
* Sample results
```
Results in milliseconds:
quickSort in JS:    2184,2134,2089,2082,2070,2048,2227,2204,2009,2056,2020,2141,2050,2150,2095,2261, average: 2113
quickSort in Wasm:  390,370,372,400,374,364,362,372,379,370,397,363,370,399,368,363, average: 375
Array.sort() in V8: 145,145,143,143,146,146,143,144,146,145,144,144,145,144,143,145, average: 144 // Google's dark art ¯\_(ツ)_/¯ (explained: https://v8.dev/blog/array-sort)
quickSort in C:     176,165,161,162,162,163,163,161,163,164,162,195,164,165,162,176, avg: 166
```