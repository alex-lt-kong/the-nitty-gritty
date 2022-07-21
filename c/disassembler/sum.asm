0x0000000000000000 <+0>:	push   rbp                          ; rbp is the frame pointer on x86_64. here we save address of previous stack frame
0x0000000000000001 <+1>:	mov    rbp,rsp                      ; rsp is the current stack pointer: ; Address of current stack frame
0x0000000000000004 <+4>:	mov    DWORD PTR [rbp-0x14],edi     ; move 1st argument, int a, into a pointer offset address
0x0000000000000007 <+7>:	mov    DWORD PTR [rbp-0x18],esi     ; move 2nd argument, int b, into a pointer offset address
0x000000000000000a <+10>:	mov    DWORD PTR [rbp-0x4],0x0      ; int temp = 0;
0x0000000000000011 <+17>:	mov    edx,DWORD PTR [rbp-0x14]     ; move int a into edx
0x0000000000000014 <+20>:	mov    eax,DWORD PTR [rbp-0x18]     ; move int b into eax
0x0000000000000017 <+23>:	add    eax,edx                      ; add value from edx to eax
0x0000000000000019 <+25>:	mov    DWORD PTR [rbp-0x4],eax      ; move the sum from eax to temp
0x000000000000001c <+28>:	mov    eax,DWORD PTR [rbp-0x4]      ; ? why we move back?
0x000000000000001f <+31>:	pop    rbp                          ; return temp;
0x0000000000000020 <+32>:	ret    
