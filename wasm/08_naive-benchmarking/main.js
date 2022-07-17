const Module = require('./func.js');
const wasm = Module({wasmBinaryFile: 'func.wasm'});

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


function quickSort(array) {
  if (array.length <= 1) {
    return array;
  }

  var pivot = array[0];
  
  var left = []; 
  var right = [];

  for (var i = 1; i < array.length; i++) {
    array[i] < pivot ? left.push(array[i]) : right.push(array[i]);
  }

  return quickSort(left).concat(pivot, quickSort(right));
};

results = {
  v8: [],
  wasm: [],
  js: []
}
const ITER = 16;
const SIZE = 2 * 1024 * 1024;
let inArray = new Int32Array(SIZE);
for (let i = 0; i < ITER; ++i) {
  for (let j = 0; j < SIZE; ++j) {
      inArray[j] = Math.round(Math.random() * 2147483647);
  }
  console.log(`${i + 1}-th iteration...`);
  let startTime = (new Date()).getTime();
  
  const inArrayPtr = arrayToPtr(inArray);
  const outArrayWasm = ptrToArray(wasm._quick_sort(inArrayPtr, 0, inArray.length-1), inArray.length);
  
  wasm._free(inArrayPtr);
  results.wasm.push((new Date()).getTime() - startTime);

  startTime = (new Date()).getTime();
  const outArrayJs = quickSort(inArray);
  results.js.push((new Date()).getTime() - startTime);

  startTime = (new Date()).getTime();
  inArray.sort();
  results.v8.push((new Date()).getTime() - startTime);
  for (let j = 0; j < SIZE; ++j) {
    if (outArrayJs[j] !== outArrayWasm[j] || outArrayWasm[j] !== inArray[j]) {
      console.error("Sort results not identical");
    }
  }
}
const average = (array) => array.reduce((a, b) => a + b) / array.length;
console.log(`Results in milliseconds:`);
console.log(`quickSort in JS:    ${results.js}, average: ${average(results.js)}`);
console.log(`quickSort in Wasm:  ${results.wasm}, average: ${average(results.wasm)}`);
console.log(`Array.sort() in V8: ${results.v8}, average: ${average(results.v8)}`);