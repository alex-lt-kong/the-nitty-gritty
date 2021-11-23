global _start

_start:
  call my_func
  mov eax, 1
  mov ebx, 0
  int 0x80

my_func:
  ; the design in 3_base-pointer-register.asm has a big issue:
  ; suppose there is a nested function call, the second call
  ; will also move esp (i.e., top of stack) to ebp, overwritting 
  ; the original ebp we want to keep. To solve this, we push the value 
  ; in ebp to the stack to preserve it. We will pop it before the function
  ; returns
  ; the below three lines are called the prologue of a function
  push ebp
  mov ebp, esp
  sub esp, 2


  mov [esp], byte 'H'
  mov [esp+1], byte 'i'

  mov eax, 4  ; sys_write system call
  mov ebx, 1  ; stdout file descriptor
  mov ecx, esp,
  mov edx, 2  ; number of bytes to write
  int 0x80


  ; first we restore the very latest ebp back to esp (top of stack)
  ; and then we can safely pop ebp
  ; these lines are also called the epilogue of a function
  mov esp, ebp
  pop ebp ; this means we pop the stack top to ebp
  ret
  ; recall that ret is just an alias of pop and jmp
