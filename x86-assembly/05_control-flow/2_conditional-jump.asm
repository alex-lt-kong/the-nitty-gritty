global _start

section .text
_start:
  mov eax, 1        ; sys_exit system call
  mov ecx, 101  
  cmp ecx, 100
  jl my_label   ; jump if less than
  ; other common conditional jump instructions include: jg and je.
  mov ebx, 24   ; this WILL be executed
  int 0x80

my_label:

  mov ebx, 42   ; this won't be executed
  int 0x80