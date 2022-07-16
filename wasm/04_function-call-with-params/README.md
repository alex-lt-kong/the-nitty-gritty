* `emcc main.c -s WASM=1 -s EXPORTED_FUNCTIONS='['_wCount', '_main', '_greetX']'  -o main.js`
* Original code from [here](https://github.com/enarx/outreachy/tree/main/ajay/callingCprogramsFunction)