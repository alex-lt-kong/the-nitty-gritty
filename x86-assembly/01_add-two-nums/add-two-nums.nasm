global _start
_start:
  mov eax, 1
  mov ebx, 42
  add ebx, 3
  int 0x80
  /* 
  int means interrupt, and the number 0x80 is the interrupt number.
  An interrupt transfers the program flow to whomever is handling that interrupt, which is interrupt 0x80 in this case.
  The kernel is notified about which system call the program wants to make, by examining the value in the register eax (Intel syntax).
  Each system call have different requirements about the use of the other registers.
  1 in eax means we want to make a system exit call and the value stored in ebx
  will be the exit code for the program
  */