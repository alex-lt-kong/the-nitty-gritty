Dump of assembler code for function floating_division_aos:
21	void floating_division_aos(float a, float b, float c, struct pixel** arr, struct pixel** results, size_t arr_len) {
   0x0000000000000000 <+0>:	movaps xmm3,xmm0
   0x0000000000000003 <+3>:	mov    r8,rsi

22	  
23	  #if defined( __INTEL_COMPILER)
24	  #pragma ivdep
25	  // Pragmas are specific for the compiler and platform in use. So the best bet is to look at compiler's documentation.
26	  // https://stackoverflow.com/questions/5078679/what-is-the-scope-of-a-pragma-directive
27	  #elif defined(__GNUC__)
28	  #pragma GCC ivdep
29	  #endif
30	  for (int i = 0; i < arr_len; ++i) {
   0x0000000000000006 <+6>:	test   rdx,rdx
   0x0000000000000009 <+9>:	je     0x50 <floating_division_aos+80>
   0x000000000000000b <+11>:	lea    rsi,[rdx*8+0x0]
   0x0000000000000013 <+19>:	xor    eax,eax
   0x0000000000000015 <+21>:	nop    DWORD PTR [rax]
   0x0000000000000020 <+32>:	add    rax,0x8
   0x000000000000004b <+75>:	cmp    rsi,rax
   0x000000000000004e <+78>:	jne    0x18 <floating_division_aos+24>

31	    results[i]->r = arr[i]->r / a;
   0x0000000000000018 <+24>:	mov    rcx,QWORD PTR [rdi+rax*1]
   0x000000000000001c <+28>:	mov    rdx,QWORD PTR [r8+rax*1]
   0x0000000000000024 <+36>:	movss  xmm0,DWORD PTR [rcx]
   0x0000000000000028 <+40>:	divss  xmm0,xmm3
   0x000000000000002c <+44>:	movss  DWORD PTR [rdx],xmm0

32	    results[i]->g = b / arr[i]->g;
   0x0000000000000030 <+48>:	movaps xmm0,xmm1
   0x0000000000000033 <+51>:	divss  xmm0,DWORD PTR [rcx+0x4]
   0x0000000000000038 <+56>:	movss  DWORD PTR [rdx+0x4],xmm0

33	    results[i]->b = arr[i]->b / c;
   0x000000000000003d <+61>:	movss  xmm0,DWORD PTR [rcx+0x8]
   0x0000000000000042 <+66>:	divss  xmm0,xmm2
   0x0000000000000046 <+70>:	movss  DWORD PTR [rdx+0x8],xmm0

34	  }
35	}
   0x0000000000000050 <+80>:	ret    
   0x0000000000000051:	data16 nop WORD PTR cs:[rax+rax*1+0x0]
   0x000000000000005c:	nop    DWORD PTR [rax+0x0]

End of assembler dump.
Dump of assembler code for function floating_division_soa:
37	void floating_division_soa(float a, float b, float c, struct pixelArray* arr, struct pixelArray* results, size_t arr_len) {
   0x0000000000000060 <+0>:	movaps xmm3,xmm0
   0x0000000000000063 <+3>:	mov    rcx,rdi
   0x0000000000000066 <+6>:	mov    rax,rsi

38	  #if defined( __INTEL_COMPILER)
39	  #pragma ivdep
40	  #elif defined(__GNUC__)
41	  #pragma GCC ivdep
42	  #endif
43	  for (int i = 0; i < arr_len; ++i) {
   0x0000000000000069 <+9>:	test   rdx,rdx
   0x000000000000006c <+12>:	je     0xc5 <floating_division_soa+101>
   0x00000000000000bc <+92>:	add    rax,0x4
   0x00000000000000c0 <+96>:	cmp    rdx,rax
   0x00000000000000c3 <+99>:	jne    0x90 <floating_division_soa+48>

44	    results->r[i] = arr->r[i] / a;
   0x000000000000006e <+14>:	mov    r10,QWORD PTR [rdi]
   0x0000000000000071 <+17>:	mov    r9,QWORD PTR [rsi]
   0x0000000000000074 <+20>:	shl    rdx,0x2
   0x0000000000000090 <+48>:	movss  xmm0,DWORD PTR [r10+rax*1]
   0x0000000000000096 <+54>:	divss  xmm0,xmm3
   0x000000000000009a <+58>:	movss  DWORD PTR [r9+rax*1],xmm0

45	    results->g[i] = b / arr->g[i];
   0x0000000000000078 <+24>:	mov    r8,QWORD PTR [rdi+0x8]
   0x000000000000007c <+28>:	mov    rdi,QWORD PTR [rsi+0x8]
   0x00000000000000a0 <+64>:	movaps xmm0,xmm1
   0x00000000000000a3 <+67>:	divss  xmm0,DWORD PTR [r8+rax*1]
   0x00000000000000a9 <+73>:	movss  DWORD PTR [rdi+rax*1],xmm0

46	    results->b[i] = arr->b[i] / c;
   0x0000000000000080 <+32>:	mov    rsi,QWORD PTR [rcx+0x10]
   0x0000000000000084 <+36>:	mov    rcx,QWORD PTR [rax+0x10]
   0x0000000000000088 <+40>:	xor    eax,eax
   0x000000000000008a <+42>:	nop    WORD PTR [rax+rax*1+0x0]
   0x00000000000000ae <+78>:	movss  xmm0,DWORD PTR [rsi+rax*1]
   0x00000000000000b3 <+83>:	divss  xmm0,xmm2
   0x00000000000000b7 <+87>:	movss  DWORD PTR [rcx+rax*1],xmm0

47	  }
48	}
   0x00000000000000c5 <+101>:	ret    

End of assembler dump.
