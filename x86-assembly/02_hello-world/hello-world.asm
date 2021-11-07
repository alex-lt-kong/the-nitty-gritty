global _start
section .data
  my_msg db "Hello world!", 0x0a
  ; db: define byte
  ; 0x0a is the code for newline character
  msg_len equ $ - my_msg
  ; calculate the length of string and store it into msg_len
section .text
_start:
  mov eax, 4        ; sys_write system call
  mov ebx, 1        ; stdout file descriptor
  mov ecx, my_msg   ; bytes to write
  mov edx, msg_len  ; number of bytes to write
  int 0x80
  mov eax, 1        ; sys_exit system call
  mov ebx, 0        ; exit status is 0
  int 0x80