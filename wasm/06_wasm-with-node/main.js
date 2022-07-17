const Module = require('./func.js');
const wasm = Module({wasmBinaryFile: 'func.wasm'});
console.log(wasm._add(40, 40));

const sizeOfInt32 = 4;
  // Takes an Int32Array, copies it to the heap and returns a pointer
function arrayToPtr(array) {
  const ptr = wasm._my_malloc(array.length * sizeOfInt32); // _malloc() is a function exposed from wasm side.
  wasm.HEAP32.set(array, ptr / sizeOfInt32);
  // HEAP32 is a variable from Emscripten's memory model:
  // https://emscripten.org/docs/porting/emscripten-runtime-environment.html#emscripten-memory-model
  return ptr; // typeof ptr === 'number';
}

// Takes a pointer and  array length, and returns a Int32Array from the heap
function ptrToArray(ptr, length) {
  let array = new Int32Array(length);
  const pos = ptr / sizeOfInt32;
  array.set(wasm.HEAP32.subarray(pos, pos + length));
  return array;
}


//  ccall(name of C function, return type, argument types, arguments);

let inArray = new Int32Array([22, 3, 9527, 44, -123, 65536, 0, -666, 66, 999]);
console.log(`inArray: ${inArray}`);
const outArray = ptrToArray(wasm._bubble_sort(arrayToPtr(inArray), inArray.length), inArray.length);
console.log(`outArray: ${outArray}`);
