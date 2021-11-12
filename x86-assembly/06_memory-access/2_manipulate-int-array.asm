global _start

section .data
  arr db 1, 2, 3, 4, 5, 6, 7, 8, 9, 10  ; db: define byte
  ; pt is a variable, which is essentially a pointer pointing to the memory that contains the string "yellow"

section .text
_start:
  mov ebx, 1
  mov ecx, 6    ; number of iterations
my_loop:

  mov eax, 4
  mov ebx, 1
  mov ecx, [pt + ecx]
  mov edx, 6
  int 0x80

  dec ecx       ; equivalent to `sub ecx, 1` but more efficient
  cmp ecx, 0
  jg my_loop
  mov eax, 1    ; sys_exit system call
  int 0x80