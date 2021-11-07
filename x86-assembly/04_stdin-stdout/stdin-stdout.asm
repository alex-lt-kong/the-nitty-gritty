section  .data
userMsg db 'Please enter a number: '
lenUserMsg equ $ - userMsg             ;The length of the message
dispMsg db 'You have entered: '
lenDispMsg equ $ - dispMsg                 

section .bss            ;Uninitialized data
num resb 5
section .text           ;Code Segment
   global _start
_start:
   ;User prompt
   mov eax, 4
   mov ebx, 1
   mov ecx, userMsg
   mov edx, lenUserMsg
   int 0x80

   ;Read and store the user input
   mov eax, 3
   mov ebx, 2
   mov ecx, num  
   mov edx, 5       ;5 bytes (numeric, 1 for sign) of that information
   int 0x80

   ;Output the message 'The entered number is: '
   mov eax, 4
   mov ebx, 1
   mov ecx, dispMsg
   mov edx, lenDispMsg
   int 0x80  

   ;Output the number entered
   mov eax, 4
   mov ebx, 1
   mov ecx, num
   mov edx, 5
   int 0x80
  
   ; Exit code
   mov eax, 1
   mov ebx, 0
   int 0x80