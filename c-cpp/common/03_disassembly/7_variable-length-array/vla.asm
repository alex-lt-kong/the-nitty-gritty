
./vla.o:     file format elf64-x86-64


Disassembly of section .text:

0000000000000000 <print_vla>:
   0:	push   rbp
   1:	mov    rbp,rsp
   4:	push   rbx
   5:	sub    rsp,0x38
   9:	mov    QWORD PTR [rbp-0x38],rdi
   d:	mov    rax,rsp
  10:	mov    rbx,rax
  13:	mov    rax,QWORD PTR [rbp-0x38]
  17:	mov    rdx,rax
  1a:	sub    rdx,0x1
  1e:	mov    QWORD PTR [rbp-0x28],rdx
  22:	mov    r10,rax
  25:	mov    r11d,0x0
  2b:	mov    r8,rax
  2e:	mov    r9d,0x0
  34:	lea    rdx,[rax*4+0x0]
  3c:	mov    eax,0x10
  41:	sub    rax,0x1
  45:	add    rax,rdx
  48:	mov    esi,0x10
  4d:	mov    edx,0x0
  52:	div    rsi
  55:	imul   rax,rax,0x10
  59:	sub    rsp,rax
  5c:	mov    rax,rsp
  5f:	add    rax,0x3
  63:	shr    rax,0x2
  67:	shl    rax,0x2
  6b:	mov    QWORD PTR [rbp-0x30],rax
  6f:	mov    rax,QWORD PTR [rbp-0x30]
  73:	mov    DWORD PTR [rax],0x0
  79:	mov    rax,QWORD PTR [rbp-0x30]
  7d:	mov    DWORD PTR [rax+0x4],0x1
  84:	mov    QWORD PTR [rbp-0x20],0x2
  8c:	jmp    be <print_vla+0xbe>
  8e:	mov    rax,QWORD PTR [rbp-0x20]
  92:	lea    rdx,[rax-0x1]
  96:	mov    rax,QWORD PTR [rbp-0x30]
  9a:	mov    ecx,DWORD PTR [rax+rdx*4]
  9d:	mov    rax,QWORD PTR [rbp-0x20]
  a1:	lea    rdx,[rax-0x2]
  a5:	mov    rax,QWORD PTR [rbp-0x30]
  a9:	mov    eax,DWORD PTR [rax+rdx*4]
  ac:	add    ecx,eax
  ae:	mov    rax,QWORD PTR [rbp-0x30]
  b2:	mov    rdx,QWORD PTR [rbp-0x20]
  b6:	mov    DWORD PTR [rax+rdx*4],ecx
  b9:	add    QWORD PTR [rbp-0x20],0x1
  be:	mov    rax,QWORD PTR [rbp-0x20]
  c2:	cmp    rax,QWORD PTR [rbp-0x38]
  c6:	jb     8e <print_vla+0x8e>
  c8:	mov    QWORD PTR [rbp-0x18],0x0
  d0:	jmp    f5 <print_vla+0xf5>
  d2:	mov    rax,QWORD PTR [rbp-0x30]
  d6:	mov    rdx,QWORD PTR [rbp-0x18]
  da:	mov    eax,DWORD PTR [rax+rdx*4]
  dd:	mov    esi,eax
  df:	lea    rdi,[rip+0x0]        # e6 <print_vla+0xe6>
  e6:	mov    eax,0x0
  eb:	call   f0 <print_vla+0xf0>
  f0:	add    QWORD PTR [rbp-0x18],0x1
  f5:	mov    rax,QWORD PTR [rbp-0x18]
  f9:	cmp    rax,QWORD PTR [rbp-0x38]
  fd:	jb     d2 <print_vla+0xd2>
  ff:	mov    edi,0xa
 104:	call   109 <print_vla+0x109>
 109:	nop
 10a:	mov    rsp,rbx
 10d:	mov    rbx,QWORD PTR [rbp-0x8]
 111:	leave  
 112:	ret    
