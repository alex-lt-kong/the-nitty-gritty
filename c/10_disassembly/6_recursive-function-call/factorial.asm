
./factorial.o:     file format elf64-x86-64


Disassembly of section .text:

0000000000000000 <factorial>:
   0:	push   rbp
   1:	mov    rbp,rsp
   4:	sub    rsp,0x10
   8:	mov    QWORD PTR [rbp-0x8],rdi
   c:	cmp    QWORD PTR [rbp-0x8],0x1
  11:	jne    19 <factorial+0x19>
  13:	mov    rax,QWORD PTR [rbp-0x8]
  17:	jmp    2e <factorial+0x2e>
  19:	mov    rax,QWORD PTR [rbp-0x8]
  1d:	sub    rax,0x1
  21:	mov    rdi,rax
  24:	call   29 <factorial+0x29>
  29:	imul   rax,QWORD PTR [rbp-0x8]
  2e:	leave  
  2f:	ret    
