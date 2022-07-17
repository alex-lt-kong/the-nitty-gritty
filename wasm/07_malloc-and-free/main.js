const Module = require('./func.js');
const wasm = Module({wasmBinaryFile: 'func.wasm'});
console.log(wasm._add(40, 40));

const sizeOfInt32 = 4;

function arrayToPtr(array) {
  const ptr = wasm._malloc(array.length * sizeOfInt32); // _malloc() is a function exposed from wasm side.
  wasm.HEAP32.set(array, ptr / sizeOfInt32);
  return ptr; // typeof ptr === 'number';
}

function ptrToArray(ptr, length) {
  let array = new Int32Array(length);
  const pos = ptr / sizeOfInt32;
  array.set(wasm.HEAP32.subarray(pos, pos + length));
  return array;
}

const ITER = 65536;
for (let i = 0; i < ITER; ++i) {
  const SIZE = 1024;
  let inArray = new Int32Array(SIZE);
  for (let j = 0; j < SIZE; ++j) {
      inArray[j] = Math.round(Math.random() * 2147483647);
  }
  console.log(`${i}-th iteration`);
  console.log(`inArray: ${inArray.slice(0, 10)}`);
  const inArrayPtr = arrayToPtr(inArray);
  const outArray = ptrToArray(wasm._bubble_sort(inArrayPtr, inArray.length), inArray.length);
  console.log(`outArray: ${outArray.slice(0, 10)}`); 
  wasm._free(inArrayPtr); // withOUT this statement, the script throws OOM exception after a short while.
}
