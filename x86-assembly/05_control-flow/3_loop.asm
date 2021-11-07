global _start

section .text
_start:
  mov ebx, 1
  mov ecx, 6    ; number of iterations
my_loop:
  add ebx, ebx  ; ebx += ebx
  dec ecx       ; equivalent to `sub ecx, 1` but more efficient
  cmp ecx, 0
  jg my_loop
  mov eax, 1    ; sys_exit system call
  int 0x80