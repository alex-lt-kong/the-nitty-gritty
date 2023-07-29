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
   0x0000000000000060 <+0>:	movaps xmm6,xmm0
   0x0000000000000063 <+3>:	mov    rcx,rdi
   0x0000000000000066 <+6>:	mov    rax,rsi

38	  #if defined( __INTEL_COMPILER)
39	  #pragma ivdep
40	  #elif defined(__GNUC__)
41	  #pragma GCC ivdep
42	  #endif
43	  for (int i = 0; i < arr_len; ++i) {
   0x0000000000000069 <+9>:	test   rdx,rdx
   0x000000000000006c <+12>:	je     0x1a0 <floating_division_soa+320>
   0x00000000000000e7 <+135>:	add    rax,0x10
   0x00000000000000eb <+139>:	cmp    rax,r11
   0x00000000000000ee <+142>:	jne    0xc0 <floating_division_soa+96>
   0x00000000000000f0 <+144>:	mov    r11,rdx
   0x00000000000000f3 <+147>:	and    r11,0xfffffffffffffffc
   0x00000000000000f7 <+151>:	mov    eax,r11d
   0x00000000000000fa <+154>:	cmp    rdx,r11
   0x00000000000000fd <+157>:	je     0x1a8 <floating_division_soa+328>
   0x0000000000000132 <+210>:	lea    r11d,[rax+0x1]
   0x0000000000000136 <+214>:	movsxd r11,r11d
   0x0000000000000139 <+217>:	cmp    rdx,r11
   0x000000000000013c <+220>:	jbe    0x1a0 <floating_division_soa+320>
   0x0000000000000144 <+228>:	add    eax,0x2
   0x0000000000000147 <+231>:	cdqe   
   0x0000000000000172 <+274>:	cmp    rdx,rax
   0x0000000000000175 <+277>:	jbe    0x1a0 <floating_division_soa+320>
   0x00000000000001a9 <+329>:	xor    eax,eax
   0x00000000000001ab <+331>:	xor    r11d,r11d
   0x00000000000001ae <+334>:	jmp    0x103 <floating_division_soa+163>

44	    results->r[i] = arr->r[i] / a;
   0x0000000000000072 <+18>:	mov    r10,QWORD PTR [rdi]
   0x0000000000000075 <+21>:	mov    r9,QWORD PTR [rsi]
   0x00000000000000c0 <+96>:	movups xmm0,XMMWORD PTR [r10+rax*1]
   0x00000000000000c5 <+101>:	divps  xmm0,xmm5
   0x00000000000000c8 <+104>:	movups XMMWORD PTR [r9+rax*1],xmm0
   0x0000000000000103 <+163>:	movss  xmm0,DWORD PTR [r10+r11*4]
   0x0000000000000109 <+169>:	divss  xmm0,xmm6
   0x000000000000010d <+173>:	movss  DWORD PTR [r9+r11*4],xmm0
   0x000000000000013e <+222>:	movss  xmm0,DWORD PTR [r10+r11*4]
   0x0000000000000149 <+233>:	divss  xmm0,xmm6
   0x000000000000014d <+237>:	movss  DWORD PTR [r9+r11*4],xmm0
   0x0000000000000177 <+279>:	movss  xmm0,DWORD PTR [r10+rax*4]
   0x000000000000017d <+285>:	divss  xmm0,xmm6
   0x0000000000000181 <+289>:	movss  DWORD PTR [r9+rax*4],xmm0

45	    results->g[i] = b / arr->g[i];
   0x0000000000000078 <+24>:	mov    r8,QWORD PTR [rdi+0x8]
   0x000000000000007c <+28>:	mov    rdi,QWORD PTR [rsi+0x8]
   0x00000000000000cd <+109>:	movups xmm7,XMMWORD PTR [r8+rax*1]
   0x00000000000000d2 <+114>:	movaps xmm0,xmm4
   0x00000000000000d5 <+117>:	divps  xmm0,xmm7
   0x00000000000000d8 <+120>:	movups XMMWORD PTR [rdi+rax*1],xmm0
   0x0000000000000113 <+179>:	movaps xmm0,xmm1
   0x0000000000000116 <+182>:	divss  xmm0,DWORD PTR [r8+r11*4]
   0x000000000000011c <+188>:	movss  DWORD PTR [rdi+r11*4],xmm0
   0x0000000000000153 <+243>:	movaps xmm0,xmm1
   0x0000000000000156 <+246>:	divss  xmm0,DWORD PTR [r8+r11*4]
   0x000000000000015c <+252>:	movss  DWORD PTR [rdi+r11*4],xmm0
   0x0000000000000187 <+295>:	divss  xmm1,DWORD PTR [r8+rax*4]
   0x000000000000018d <+301>:	movss  DWORD PTR [rdi+rax*4],xmm1

46	    results->b[i] = arr->b[i] / c;
   0x0000000000000080 <+32>:	mov    rsi,QWORD PTR [rcx+0x10]
   0x0000000000000084 <+36>:	mov    rcx,QWORD PTR [rax+0x10]
   0x0000000000000088 <+40>:	lea    rax,[rdx-0x1]
   0x000000000000008c <+44>:	cmp    rax,0x2
   0x0000000000000090 <+48>:	jbe    0x1a9 <floating_division_soa+329>
   0x0000000000000096 <+54>:	mov    r11,rdx
   0x0000000000000099 <+57>:	movaps xmm5,xmm0
   0x000000000000009c <+60>:	movaps xmm4,xmm1
   0x000000000000009f <+63>:	xor    eax,eax
   0x00000000000000a1 <+65>:	shr    r11,0x2
   0x00000000000000a5 <+69>:	movaps xmm3,xmm2
   0x00000000000000a8 <+72>:	shufps xmm5,xmm5,0x0
   0x00000000000000ac <+76>:	shufps xmm4,xmm4,0x0
   0x00000000000000b0 <+80>:	shl    r11,0x4
   0x00000000000000b4 <+84>:	shufps xmm3,xmm3,0x0
   0x00000000000000b8 <+88>:	nop    DWORD PTR [rax+rax*1+0x0]
   0x00000000000000dc <+124>:	movups xmm0,XMMWORD PTR [rsi+rax*1]
   0x00000000000000e0 <+128>:	divps  xmm0,xmm3
   0x00000000000000e3 <+131>:	movups XMMWORD PTR [rcx+rax*1],xmm0
   0x0000000000000122 <+194>:	movss  xmm0,DWORD PTR [rsi+r11*4]
   0x0000000000000128 <+200>:	divss  xmm0,xmm2
   0x000000000000012c <+204>:	movss  DWORD PTR [rcx+r11*4],xmm0
   0x0000000000000162 <+258>:	movss  xmm0,DWORD PTR [rsi+r11*4]
   0x0000000000000168 <+264>:	divss  xmm0,xmm2
   0x000000000000016c <+268>:	movss  DWORD PTR [rcx+r11*4],xmm0
   0x0000000000000192 <+306>:	movss  xmm0,DWORD PTR [rsi+rax*4]
   0x0000000000000197 <+311>:	divss  xmm0,xmm2
   0x000000000000019b <+315>:	movss  DWORD PTR [rcx+rax*4],xmm0

47	  }
48	}
   0x00000000000001a0 <+320>:	ret    
   0x00000000000001a1 <+321>:	nop    DWORD PTR [rax+0x0]
   0x00000000000001a8 <+328>:	ret    

End of assembler dump.
