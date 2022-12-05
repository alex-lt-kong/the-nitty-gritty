# sizeof() a pointer

* Without thinking it a bit deeper, it may be tempting to think that sizeof() a pointer is related to the type it 
points to. Using `uint8_t* ptr8` and `int32_t* ptr32` as example, one may assume that `sizeof(ptr32)` is four
times larger than `sizeof(ptr8)`.

* This is a misconception...As pointer stores the address of another variable, it should be always as long as the
length of memory address, a.k.a., the Word size of the processor.
  * So for a 32-bit CPU, a pointer is always 32-bit long and for a 64-bit CPU a pointer is always 64-bit long.
  * Even if a pointer points to a 8-bit byte, it still has to be 32/64-bit long.
    