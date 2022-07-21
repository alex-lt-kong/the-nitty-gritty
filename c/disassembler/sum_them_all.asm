0x0000000000000021 <+0>:	push   rbp
0x0000000000000022 <+1>:	mov    rbp,rsp
0x0000000000000025 <+4>:	mov    QWORD PTR [rbp-0x18],rdi  ; move 1st argument, int* arr, into a pointer offset address
0x0000000000000029 <+8>:	mov    DWORD PTR [rbp-0x1c],esi  ; move 2nd argument, int arr_len, into a pointer offset address
0x000000000000002c <+11>:	mov    DWORD PTR [rbp-0x4],0x0   ; int sum = 0: set 32-bit integer at address [rbp-0x4] to 0
0x0000000000000033 <+18>:	mov    DWORD PTR [rbp-0x8],0x0   ; int i = 0: set 32-bit integer at address [rbp-0x8] to 0
0x000000000000003a <+25>:	jmp    0x59 <sum_them_all+56>    ; jump to 0x0000000000000059, i.e., <+56>
; start of sum += arr[i];
0x000000000000003c <+27>:	mov    eax,DWORD PTR [rbp-0x8]   ; move value from 
0x000000000000003f <+30>:	cdqe   
; cdqe: Convert Double to Quad Extend; AT&T equivalent is cltq: Convert Long To Quad;
; quad-word is 8-byte long while long/double-word is 4-byte long
; this instruction applies to rax register
0x0000000000000041 <+32>:	lea    rdx,[rax*4+0x0]
; lea: load effective address
0x0000000000000049 <+40>:	mov    rax,QWORD PTR [rbp-0x18]
0x000000000000004d <+44>:	add    rax,rdx
0x0000000000000050 <+47>:	mov    eax,DWORD PTR [rax]
0x0000000000000052 <+49>:	add    DWORD PTR [rbp-0x4],eax
; end of sum += arr[i];
0x0000000000000055 <+52>:	add    DWORD PTR [rbp-0x8],0x1
0x0000000000000059 <+56>:	mov    eax,DWORD PTR [rbp-0x8]   ; move [rbp-0x8] (i.e., i) to eax;
0x000000000000005c <+59>:	cmp    eax,DWORD PTR [rbp-0x1c]  ; i < arr_len: compare eax (i.e., i) to [rbp-0x1c] (i.e., arr_len)
0x000000000000005f <+62>:	jl     0x3c <sum_them_all+27>    ; i < arr_len: Jump to 0x000000000000003c, i.e., <+27> if eax (i.e., i) is Less than [rbp-0x1c] (i.e., arr_len)
0x0000000000000061 <+64>:	mov    eax,DWORD PTR [rbp-0x4]
0x0000000000000064 <+67>:	pop    rbp
0x0000000000000065 <+68>:	ret    
