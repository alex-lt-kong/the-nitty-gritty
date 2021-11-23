global _start

_start:
  call my_func
  mov eax, 1
  mov ebx, 0
  int 0x80

my_func:
  mov ebp, esp
  ; ebp is the base pointer register
  ; esp is the register holding the value pointing to the top of the stack
  ; so here we use ebp to store the current value in esp,
  ; i.e., the location of the top of the stack
  sub esp, 2 ; allocate two bytes on the stack
  ; since we already marked down the original top of the stack, now we
  ; are free to manipulate the stack!
  mov [esp], byte 'H'
  mov [esp+1], byte 'i'

  mov eax, 4  ; sys_write system call
  mov ebx, 1  ; stdout file descriptor
  mov ecx, esp,
  mov edx, 2  ; number of bytes to write
  int 0x80

  mov esp, ebp
  ; restore value from ebp back to esp
  ; essentially it de-allocates the space we just allocated.
  ret
  ; recall that ret is just an alias of pop and jmp