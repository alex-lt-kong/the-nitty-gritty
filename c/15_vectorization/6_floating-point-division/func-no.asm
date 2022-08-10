oid func_floating_division(float a, float b, float c, float d, float* arr, float* results, size_t arr_len) {
0x0000000000000000 <+0>:	movaps xmm5,xmm0

 for (int i = 0; i < arr_len; ++i) {
0x0000000000000003 <+3>:	test   rdx,rdx
0x0000000000000006 <+6>:	je     0x5e <func_floating_division+94>
0x0000000000000008 <+8>:	lea    rax,[rdi+rdx*4]
0x000000000000000c <+12>:	nop    DWORD PTR [rax+0x0]
0x0000000000000017 <+23>:	add    rdi,0x4
0x000000000000001b <+27>:	add    rsi,0x4
0x0000000000000059 <+89>:	cmp    rdi,rax
0x000000000000005c <+92>:	jne    0x10 <func_floating_division+16>

   results[i] =  arr[i] / a;
0x0000000000000010 <+16>:	movss  xmm4,DWORD PTR [rdi]
0x000000000000001f <+31>:	divss  xmm4,xmm5
0x0000000000000023 <+35>:	movss  DWORD PTR [rsi-0x4],xmm4

   results[i] += b / arr[i];
0x0000000000000014 <+20>:	movaps xmm0,xmm1
0x0000000000000028 <+40>:	divss  xmm0,DWORD PTR [rdi-0x4]
0x000000000000002d <+45>:	addss  xmm4,xmm0
0x0000000000000031 <+49>:	movss  DWORD PTR [rsi-0x4],xmm4

   results[i] += arr[i] / c;
0x0000000000000036 <+54>:	movss  xmm0,DWORD PTR [rdi-0x4]
0x000000000000003b <+59>:	divss  xmm0,xmm2
0x000000000000003f <+63>:	addss  xmm0,xmm4
0x0000000000000046 <+70>:	movss  DWORD PTR [rsi-0x4],xmm0

   results[i] += d / arr[i];
0x0000000000000043 <+67>:	movaps xmm4,xmm3
0x000000000000004b <+75>:	divss  xmm4,DWORD PTR [rdi-0x4]
0x0000000000000050 <+80>:	addss  xmm0,xmm4
0x0000000000000054 <+84>:	movss  DWORD PTR [rsi-0x4],xmm0

 }
}
0x000000000000005e <+94>:	ret    
0x000000000000005f:	nop

 of assembler dump.
p of assembler code for function func_int_multiplication:
void func_int_multiplication(int32_t a, int32_t b, int32_t c, int32_t d, int32_t* arr, int32_t* results, size_t arr_len) {
0x0000000000000060 <+0>:	mov    rax,QWORD PTR [rsp+0x8]
0x0000000000000065 <+5>:	mov    r10d,edx

  for (int i = 0; i < arr_len; ++i) {
0x0000000000000068 <+8>:	test   rax,rax
0x000000000000006b <+11>:	je     0xb7 <func_int_multiplication+87>
0x000000000000006d <+13>:	lea    r11,[r8+rax*4]
0x0000000000000071 <+17>:	nop    DWORD PTR [rax+0x0]
0x000000000000007b <+27>:	add    r8,0x4
0x000000000000007f <+31>:	add    r9,0x4
0x00000000000000b2 <+82>:	cmp    r8,r11
0x00000000000000b5 <+85>:	jne    0x78 <func_int_multiplication+24>

    results[i] =  arr[i] * a;
0x0000000000000078 <+24>:	mov    edx,DWORD PTR [r8]
0x0000000000000083 <+35>:	imul   edx,edi
0x0000000000000086 <+38>:	mov    DWORD PTR [r9-0x4],edx

    results[i] += arr[i] * b;
0x000000000000008a <+42>:	mov    eax,DWORD PTR [r8-0x4]
0x000000000000008e <+46>:	imul   eax,esi
0x0000000000000091 <+49>:	add    edx,eax
0x0000000000000093 <+51>:	mov    DWORD PTR [r9-0x4],edx

    results[i] += arr[i] * c;
0x0000000000000097 <+55>:	mov    eax,DWORD PTR [r8-0x4]
0x000000000000009b <+59>:	imul   eax,r10d
0x000000000000009f <+63>:	add    eax,edx
0x00000000000000a1 <+65>:	mov    DWORD PTR [r9-0x4],eax

    results[i] += arr[i] * d;
0x00000000000000a5 <+69>:	mov    edx,DWORD PTR [r8-0x4]
0x00000000000000a9 <+73>:	imul   edx,ecx
0x00000000000000ac <+76>:	add    eax,edx
0x00000000000000ae <+78>:	mov    DWORD PTR [r9-0x4],eax

  }
}
0x00000000000000b7 <+87>:	ret    

