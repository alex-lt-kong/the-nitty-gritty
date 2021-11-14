; For this code, we can only choose elf32 (for nasm) and elf_i386 (for ld),
; most likely because the default memory block size is different

global _start

section .text
_start:

  push dword 33 ; ASCII code for !

  sub esp, 4  ; allocate one memory block in the stack by moving the stack pointer esp
              ; note that in x86 assembly, the top of the stack is esp == 0
  mov [esp], byte 'w'
  mov [esp+1], byte 'o'
  mov [esp+2], byte 'r'
  mov [esp+3], byte 'd' ; Note we only push once and we write 4 bytes

  push dword "hell" ; a memory block is only 4-byte long so o is omitted lol...

  mov eax, 4  ; sys_write system_call
  mov ebx, 1  ; stdout file descriptor
  mov ecx, esp; pointer to bytes to write
  mov edx, 4  ; number of bytes to write
  int 0x80

  add esp, 4
  mov eax, 4  ; sys_write system_call
  mov ebx, 1  ; stdout file descriptor
  mov ecx, esp; pointer to bytes to write
  mov edx, 4  ; number of bytes to write
  int 0x80

  add esp, 4  ; we canNOT use pop very easily here...
  mov eax, 4  ; sys_write system_call
  mov ebx, 1  ; stdout file descriptor
  mov ecx, esp; pointer to bytes to write
  mov edx, 4  ; number of bytes to write
  int 0x80

  mov eax, 1
  mov ebx, 0
  int 0x80