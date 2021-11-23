global _start

_start:
  call my_func
  ; when we use the call operation, the location of the NEXT instruction,
  ; i.e., mov eax, 1, will be pushed into the stack. Note that call does NOT
  ; push the first instruction of my_func!
  mov eax, 1
  ; for the sys exit call, ebx is used to store the exit code, which is NOT
  ; set here but is set in my_func
  int 0x80

my_func:
  mov ebx, 42
  ret
  ; ret is an alias of pop and jmp
  ; pop ecx
  ; correspondingly, we need to pop the return location to ecx. The return 
  ; location (i.e., the location of mov eax, 1) is pushed into the stack by
  ; the call operation.
  ; jmp ecx
  ; we jump back to the location stored in eax, which is the location of the 
  ; instruction immediately after the "call" operation