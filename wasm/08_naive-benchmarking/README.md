* Wasm: `emcc func.c -s WASM=1 -Wall -s MODULARIZE=1 -s BINARYEN_ASYNC_COMPILATION=0 -s EXPORTED_FUNCTIONS="['_malloc', '_free']" -o func.js`.
* C: `gcc main.c -o main.out -O3`.
* Execute: `node main.js`.
* Sample results
```
Results in milliseconds:
quickSort in JS:    2036,2129,2114,2082,2071,2101,2184,2437,1978,2191, average: 2132.3
quickSort in Wasm:  402,367,360,392,368,363,363,365,361,365, average: 370.6
Array.sort() in V8: 142,144,143,141,142,142,173,144,144,145, average: 146 /* Google's dark art ¯\_(ツ)_/¯ */
quickSort in C:     160,158,161,159,160,162,161,162,271,164, average: 171
```