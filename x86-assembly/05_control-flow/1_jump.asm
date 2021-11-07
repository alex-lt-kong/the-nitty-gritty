global _start

section .text
_start:
  mov ebx, 42
  jmp my_label
  mov ebx, 24   ; this won't be executed

my_label:
  mov eax, 1        ; sys_exit system call
  int 0x80