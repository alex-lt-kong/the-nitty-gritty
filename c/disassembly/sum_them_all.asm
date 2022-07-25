0x0000000000000021 <+0>:	push   rbp
0x0000000000000022 <+1>:	mov    rbp,rsp
0x0000000000000025 <+4>:	mov    QWORD PTR [rbp-0x18],rdi  ; [rbp-0x18] = arr, note that [rbp-0x18] stores arr, not arr[0]
; di in rdi means Destination Index, it is used to store the 1st argument of the function call sum_them_all(int* arr, int arr_len)
; the instruction means to move 1st argument, int* arr, into an address stored in rbp-0x18, or, in C's term, *(rbp-0x18)
; Here [] is the dereference operator, the same as * in C.
; Then how about QWORD PTR? per https://stackoverflow.com/questions/33119165/is-ptr-keyword-necessary-in-intel-assembly-syntax,
; QWORD PTR is a cast operator, casting value [rbp-0x18] to a QWORD PTR. WithOUT it, perhaps [rbp-0x18] is PTR, not DWORD PTR...
0x0000000000000029 <+8>:	mov    DWORD PTR [rbp-0x1c],esi
; si in esi means Source Index, it is used to store the 2ndargument of the function call sum_them_all(int* arr, int arr_len)
; the instruction means to move 2nd argument, int arr_len, into [rbp-0x18], i.e., the memory block whose address is stored in rbp-0x18
0x000000000000002c <+11>:	mov    DWORD PTR [rbp-0x4],0x0   ; int sum = 0: set 32-bit integer at [rbp-0x4] to 0, i.e., *(rbp-0x4) = 0
0x0000000000000033 <+18>:	mov    DWORD PTR [rbp-0x8],0x0   ; int i = 0:
0x000000000000003a <+25>:	jmp    0x59 <sum_them_all+56>    ; jump to 0x0000000000000059, i.e., <+56>

; start of sum += arr[i];
0x000000000000003c <+27>:	mov    eax,DWORD PTR [rbp-0x8]   ; eax = i
0x000000000000003f <+30>:	cdqe   
; cdqe: Convert Double to Quad Extend; AT&T equivalent is cltq: Convert Long To Quad;
; quad-word is 8-byte long while long/double-word is 4-byte long
; this instruction applies to rax register
0x0000000000000041 <+32>:	lea    rdx,[rax*4+0x0]           ; Following the 2nd iteration, rax = (arr + i), GUESS [rax*4+0x0] stores offset, i.e., i, but why?
; lea: load effective address. My initial understanding is that it is similar to mov, just we can have expression
; inside []. For mov, we can't have rax*4 this kind of expression.
; the RAX register is usually used for return values in functions (but seems it is not used this way in this case)
; why rax*4 though? Guess is that it makes the address suitable for 4-byte long variable, i.e., int32
0x0000000000000049 <+40>:	mov    rax,QWORD PTR [rbp-0x18]  ; rax = arr, note that arr is pointer to arr[0]'s address
; note that rax is from [rbp-0x18], so rbp-0x18 is a pointer or pointer while [rbp-0x18] is a pointer :D
0x000000000000004d <+44>:	add    rax,rdx                   ; rax += i, i.e., rax = (arr + i), where rdx is offset, rax is arr
0x0000000000000050 <+47>:	mov    eax,DWORD PTR [rax]       ; eax = *(arr + i), i.e., arr[i]
0x0000000000000052 <+49>:	add    DWORD PTR [rbp-0x4],eax   ; sum += eax, i.e., sum += arr[i]
; end of sum += arr[i];

0x0000000000000055 <+52>:	add    DWORD PTR [rbp-0x8],0x1   ; i++;
0x0000000000000059 <+56>:	mov    eax,DWORD PTR [rbp-0x8]   ; move [rbp-0x8] (i.e., i) to eax;
0x000000000000005c <+59>:	cmp    eax,DWORD PTR [rbp-0x1c]  ; i < arr_len: compare eax (i.e., i) to [rbp-0x1c] (i.e., arr_len)
0x000000000000005f <+62>:	jl     0x3c <sum_them_all+27>    ; i < arr_len: Jump to 0x000000000000003c, i.e., <+27> if eax (i.e., i) is Less than [rbp-0x1c] (i.e., arr_len)
0x0000000000000061 <+64>:	mov    eax,DWORD PTR [rbp-0x4]
0x0000000000000064 <+67>:	pop    rbp
0x0000000000000065 <+68>:	ret    
