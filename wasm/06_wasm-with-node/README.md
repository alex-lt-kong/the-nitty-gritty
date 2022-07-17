* `emcc func.c -s WASM=1 -Wall -s MODULARIZE=1 -s BINARYEN_ASYNC_COMPILATION=0 -o func.js`.
* `BINARYEN_ASYNC_COMPILATION=0` disables asynchronous initialization, which seems to be better for Node.js's case.
* Execute: `node main.js`
