
./lib.o:     file format elf64-x86-64


Disassembly of section .text:

0000000000000000 <manipulate_inplace_int>:
#include <stddef.h>
#include <stdint.h>

void manipulate_inplace_int(int* arr, int* y, size_t arr_size) {
    for (int i = 0; i < arr_size; ++i)
   0:	test   rdx,rdx
   3:	je     21 <manipulate_inplace_int+0x21>
   5:	lea    rdx,[rdi+rdx*4]
   9:	nop    DWORD PTR [rax+0x0]
        arr[i] = *y + 42;
  10:	mov    eax,DWORD PTR [rsi]
    for (int i = 0; i < arr_size; ++i)
  12:	add    rdi,0x4
        arr[i] = *y + 42;
  16:	add    eax,0x2a
  19:	mov    DWORD PTR [rdi-0x4],eax
    for (int i = 0; i < arr_size; ++i)
  1c:	cmp    rdi,rdx
  1f:	jne    10 <manipulate_inplace_int+0x10>
}
  21:	ret    
  22:	data16 nop WORD PTR cs:[rax+rax*1+0x0]
  2d:	nop    DWORD PTR [rax]

0000000000000030 <manipulate_inplace_short>:

void manipulate_inplace_short(int* arr, int16_t* y, size_t arr_size) {
    for (int i = 0; i < arr_size; ++i)
  30:	test   rdx,rdx
  33:	je     4b <manipulate_inplace_short+0x1b>
        arr[i] = *y + 42;
  35:	movsx  eax,WORD PTR [rsi]
  38:	lea    rdx,[rdi+rdx*4]
  3c:	add    eax,0x2a
  3f:	nop
  40:	mov    DWORD PTR [rdi],eax
    for (int i = 0; i < arr_size; ++i)
  42:	add    rdi,0x4
  46:	cmp    rdi,rdx
  49:	jne    40 <manipulate_inplace_short+0x10>
}
  4b:	ret    
