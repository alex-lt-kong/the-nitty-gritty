global _start

section .data
  pt db "yellow"  ; db: define byte
  ; pt is a variable, which is essentially a pointer pointing to the memory that contains the string "yellow"

section .text
_start:
  mov eax, 4    ; sys_write system call: http://faculty.nps.edu/cseagle/assembly/sys_call.html
  mov ebx, 1    ; stdout file descriptor
  mov ecx, pt   ; bytes to write
  mov edx, 6    ; number of butes to write
  int 0x80

  mov [pt], byte 'H'
  ; move 'H' to the memory pointed by pt
  ; byte 'H' means we want to move the byte representation of H into the memory block
  ; keyword "byte" is important since we can also move values that are larger than byte into the memory blocks
  mov eax, 4
  mov ebx, 1
  mov ecx, pt
  mov edx, 6
  int 0x80

  mov [pt+5], byte '!'
  mov eax, 4
  mov ebx, 1
  mov ecx, pt
  mov edx, 6
  int 0x80


  mov eax, 1
  mov ebx, 0
  int 0x80