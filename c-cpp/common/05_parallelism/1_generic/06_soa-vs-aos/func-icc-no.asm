Dump of assembler code for function floating_division_aos:
21	void floating_division_aos(float a, float b, float c, struct pixel** arr, struct pixel** results, size_t arr_len) {

22	  
23	  #if defined( __INTEL_COMPILER)
24	  #pragma ivdep
25	  // Pragmas are specific for the compiler and platform in use. So the best bet is to look at compiler's documentation.
26	  // https://stackoverflow.com/questions/5078679/what-is-the-scope-of-a-pragma-directive
27	  #elif defined(__GNUC__)
28	  #pragma GCC ivdep
29	  #endif
30	  for (int i = 0; i < arr_len; ++i) {
   0x0000000000000000 <+0>:	xor    eax,eax
   0x0000000000000002 <+2>:	test   rdx,rdx
   0x0000000000000005 <+5>:	jbe    0x41 <floating_division_aos+65>
   0x0000000000000012 <+18>:	inc    rax
   0x000000000000003c <+60>:	cmp    rax,rdx
   0x000000000000003f <+63>:	jb     0x7 <floating_division_aos+7>

31	    results[i]->r = arr[i]->r / a;
   0x0000000000000007 <+7>:	mov    rcx,QWORD PTR [rdi+rax*8]
   0x000000000000000e <+14>:	mov    r8,QWORD PTR [rsi+rax*8]
   0x0000000000000015 <+21>:	movss  xmm3,DWORD PTR [rcx]
   0x0000000000000023 <+35>:	divss  xmm3,xmm0
   0x000000000000002b <+43>:	movss  DWORD PTR [r8],xmm3

32	    results[i]->g = b / arr[i]->g;
   0x000000000000000b <+11>:	movaps xmm4,xmm1
   0x000000000000001e <+30>:	divss  xmm4,DWORD PTR [rcx+0x4]
   0x0000000000000030 <+48>:	movss  DWORD PTR [r8+0x4],xmm4

33	    results[i]->b = arr[i]->b / c;
   0x0000000000000019 <+25>:	movss  xmm5,DWORD PTR [rcx+0x8]
   0x0000000000000027 <+39>:	divss  xmm5,xmm2
   0x0000000000000036 <+54>:	movss  DWORD PTR [r8+0x8],xmm5

34	  }
35	}
   0x0000000000000041 <+65>:	ret    
   0x0000000000000042 <+66>:	nop    DWORD PTR [rax+0x0]
   0x0000000000000049 <+73>:	nop    DWORD PTR [rax+0x0]

End of assembler dump.
Dump of assembler code for function floating_division_soa:
37	void floating_division_soa(float a, float b, float c, struct pixelArray* arr, struct pixelArray* results, size_t arr_len) {
   0x0000000000000050 <+0>:	push   r14
   0x000000000000005b <+11>:	mov    rax,rdx
   0x000000000000007b <+43>:	shr    rax,0x3
   0x0000000000000256 <+518>:	sub    rdx,rdi
   0x0000000000000259 <+521>:	inc    rdx
   0x000000000000025c <+524>:	dec    edx

38	  #if defined( __INTEL_COMPILER)
39	  #pragma ivdep
40	  #elif defined(__GNUC__)
41	  #pragma GCC ivdep
42	  #endif
43	  for (int i = 0; i < arr_len; ++i) {
   0x0000000000000052 <+2>:	test   rdx,rdx
   0x0000000000000055 <+5>:	jbe    0x3c8 <floating_division_soa+888>
   0x000000000000005e <+14>:	xor    ecx,ecx
   0x0000000000000076 <+38>:	mov    edi,0x1
   0x000000000000007f <+47>:	je     0x24d <floating_division_soa+509>
   0x000000000000009a <+74>:	inc    ecx
   0x000000000000023b <+491>:	cmp    rcx,rax
   0x000000000000023e <+494>:	jb     0x85 <floating_division_soa+53>
   0x0000000000000244 <+500>:	shl    ecx,0x3
   0x0000000000000247 <+503>:	movsxd rdi,ecx
   0x000000000000024a <+506>:	inc    rdi
   0x000000000000024d <+509>:	cmp    rdi,rdx
   0x0000000000000250 <+512>:	ja     0x3c8 <floating_division_soa+888>
   0x000000000000025e <+526>:	jmp    QWORD PTR [rdx*8+0x0]

44	    results->r[i] = arr->r[i] / a;
   0x0000000000000060 <+16>:	mov    r10,QWORD PTR [rsi]
   0x000000000000006b <+27>:	mov    r11,QWORD PTR [rdi]
   0x0000000000000085 <+53>:	movsxd rdi,ecx
   0x000000000000008b <+59>:	shl    rdi,0x5
   0x000000000000009c <+76>:	movss  xmm3,DWORD PTR [r11+rdi*1]
   0x00000000000000a2 <+82>:	divss  xmm3,xmm0
   0x00000000000000a6 <+86>:	movss  DWORD PTR [r10+rdi*1],xmm3
   0x00000000000000ca <+122>:	movss  xmm6,DWORD PTR [r11+rdi*1+0x4]
   0x00000000000000d1 <+129>:	divss  xmm6,xmm0
   0x00000000000000d5 <+133>:	movss  DWORD PTR [r10+rdi*1+0x4],xmm6
   0x0000000000000100 <+176>:	movss  xmm9,DWORD PTR [r11+rdi*1+0x8]
   0x0000000000000107 <+183>:	divss  xmm9,xmm0
   0x000000000000010c <+188>:	movss  DWORD PTR [r10+rdi*1+0x8],xmm9
   0x0000000000000138 <+232>:	movss  xmm12,DWORD PTR [r11+rdi*1+0xc]
   0x000000000000013f <+239>:	divss  xmm12,xmm0
   0x0000000000000144 <+244>:	movss  DWORD PTR [r10+rdi*1+0xc],xmm12
   0x0000000000000170 <+288>:	movss  xmm15,DWORD PTR [r11+rdi*1+0x10]
   0x0000000000000177 <+295>:	divss  xmm15,xmm0
   0x000000000000017c <+300>:	movss  DWORD PTR [r10+rdi*1+0x10],xmm15
   0x00000000000001a2 <+338>:	movss  xmm5,DWORD PTR [r11+rdi*1+0x14]
   0x00000000000001a9 <+345>:	divss  xmm5,xmm0
   0x00000000000001ad <+349>:	movss  DWORD PTR [r10+rdi*1+0x14],xmm5
   0x00000000000001d3 <+387>:	movss  xmm8,DWORD PTR [r11+rdi*1+0x18]
   0x00000000000001da <+394>:	divss  xmm8,xmm0
   0x00000000000001df <+399>:	movss  DWORD PTR [r10+rdi*1+0x18],xmm8
   0x0000000000000207 <+439>:	movss  xmm11,DWORD PTR [r11+rdi*1+0x1c]
   0x000000000000020e <+446>:	divss  xmm11,xmm0
   0x0000000000000213 <+451>:	movss  DWORD PTR [r10+rdi*1+0x1c],xmm11
   0x0000000000000265 <+533>:	movss  xmm3,DWORD PTR [r11+rdi*4+0x14]
   0x000000000000026f <+543>:	divss  xmm3,xmm0
   0x0000000000000273 <+547>:	movss  DWORD PTR [r10+rdi*4+0x14],xmm3
   0x0000000000000299 <+585>:	movss  xmm3,DWORD PTR [r11+rdi*4+0x10]
   0x00000000000002a3 <+595>:	divss  xmm3,xmm0
   0x00000000000002a7 <+599>:	movss  DWORD PTR [r10+rdi*4+0x10],xmm3
   0x00000000000002cd <+637>:	movss  xmm3,DWORD PTR [r11+rdi*4+0xc]
   0x00000000000002d7 <+647>:	divss  xmm3,xmm0
   0x00000000000002db <+651>:	movss  DWORD PTR [r10+rdi*4+0xc],xmm3
   0x0000000000000301 <+689>:	movss  xmm3,DWORD PTR [r11+rdi*4+0x8]
   0x000000000000030b <+699>:	divss  xmm3,xmm0
   0x000000000000030f <+703>:	movss  DWORD PTR [r10+rdi*4+0x8],xmm3
   0x0000000000000335 <+741>:	movss  xmm3,DWORD PTR [r11+rdi*4+0x4]
   0x000000000000033f <+751>:	divss  xmm3,xmm0
   0x0000000000000343 <+755>:	movss  DWORD PTR [r10+rdi*4+0x4],xmm3
   0x0000000000000369 <+793>:	movss  xmm3,DWORD PTR [r11+rdi*4]
   0x0000000000000372 <+802>:	divss  xmm3,xmm0
   0x0000000000000376 <+806>:	movss  DWORD PTR [r10+rdi*4],xmm3
   0x0000000000000397 <+839>:	movss  xmm3,DWORD PTR [r11+rdi*4-0x4]
   0x000000000000039e <+846>:	divss  xmm3,xmm0
   0x00000000000003a2 <+850>:	movss  DWORD PTR [r10+rdi*4-0x4],xmm3

45	    results->g[i] = b / arr->g[i];
   0x0000000000000063 <+19>:	mov    r14,QWORD PTR [rsi+0x8]
   0x000000000000006e <+30>:	mov    r9,QWORD PTR [rdi+0x8]
   0x0000000000000088 <+56>:	movaps xmm4,xmm1
   0x000000000000008f <+63>:	movaps xmm7,xmm1
   0x0000000000000092 <+66>:	movaps xmm10,xmm1
   0x0000000000000096 <+70>:	movaps xmm13,xmm1
   0x00000000000000ac <+92>:	movaps xmm3,xmm1
   0x00000000000000af <+95>:	divss  xmm4,DWORD PTR [r9+rdi*1]
   0x00000000000000b5 <+101>:	movss  DWORD PTR [r14+rdi*1],xmm4
   0x00000000000000dc <+140>:	movaps xmm6,xmm1
   0x00000000000000df <+143>:	divss  xmm7,DWORD PTR [r9+rdi*1+0x4]
   0x00000000000000e6 <+150>:	movss  DWORD PTR [r14+rdi*1+0x4],xmm7
   0x0000000000000113 <+195>:	movaps xmm9,xmm1
   0x0000000000000117 <+199>:	divss  xmm10,DWORD PTR [r9+rdi*1+0x8]
   0x000000000000011e <+206>:	movss  DWORD PTR [r14+rdi*1+0x8],xmm10
   0x000000000000014b <+251>:	movaps xmm12,xmm1
   0x000000000000014f <+255>:	divss  xmm13,DWORD PTR [r9+rdi*1+0xc]
   0x0000000000000156 <+262>:	movss  DWORD PTR [r14+rdi*1+0xc],xmm13
   0x0000000000000183 <+307>:	divss  xmm3,DWORD PTR [r9+rdi*1+0x10]
   0x000000000000018a <+314>:	movss  DWORD PTR [r14+rdi*1+0x10],xmm3
   0x00000000000001b4 <+356>:	divss  xmm6,DWORD PTR [r9+rdi*1+0x14]
   0x00000000000001bb <+363>:	movss  DWORD PTR [r14+rdi*1+0x14],xmm6
   0x00000000000001e6 <+406>:	divss  xmm9,DWORD PTR [r9+rdi*1+0x18]
   0x00000000000001ed <+413>:	movss  DWORD PTR [r14+rdi*1+0x18],xmm9
   0x000000000000021a <+458>:	divss  xmm12,DWORD PTR [r9+rdi*1+0x1c]
   0x0000000000000221 <+465>:	movss  DWORD PTR [r14+rdi*1+0x1c],xmm12
   0x000000000000026c <+540>:	movaps xmm4,xmm1
   0x000000000000027a <+554>:	divss  xmm4,DWORD PTR [r9+rdi*4+0x14]
   0x0000000000000281 <+561>:	movss  DWORD PTR [r14+rdi*4+0x14],xmm4
   0x00000000000002a0 <+592>:	movaps xmm4,xmm1
   0x00000000000002ae <+606>:	divss  xmm4,DWORD PTR [r9+rdi*4+0x10]
   0x00000000000002b5 <+613>:	movss  DWORD PTR [r14+rdi*4+0x10],xmm4
   0x00000000000002d4 <+644>:	movaps xmm4,xmm1
   0x00000000000002e2 <+658>:	divss  xmm4,DWORD PTR [r9+rdi*4+0xc]
   0x00000000000002e9 <+665>:	movss  DWORD PTR [r14+rdi*4+0xc],xmm4
   0x0000000000000308 <+696>:	movaps xmm4,xmm1
   0x0000000000000316 <+710>:	divss  xmm4,DWORD PTR [r9+rdi*4+0x8]
   0x000000000000031d <+717>:	movss  DWORD PTR [r14+rdi*4+0x8],xmm4
   0x000000000000033c <+748>:	movaps xmm4,xmm1
   0x000000000000034a <+762>:	divss  xmm4,DWORD PTR [r9+rdi*4+0x4]
   0x0000000000000351 <+769>:	movss  DWORD PTR [r14+rdi*4+0x4],xmm4
   0x000000000000036f <+799>:	movaps xmm4,xmm1
   0x000000000000037c <+812>:	divss  xmm4,DWORD PTR [r9+rdi*4]
   0x0000000000000382 <+818>:	movss  DWORD PTR [r14+rdi*4],xmm4
   0x00000000000003a9 <+857>:	divss  xmm1,DWORD PTR [r9+rdi*4-0x4]
   0x00000000000003b0 <+864>:	movss  DWORD PTR [r14+rdi*4-0x4],xmm1

46	    results->b[i] = arr->b[i] / c;
   0x0000000000000067 <+23>:	mov    r8,QWORD PTR [rsi+0x10]
   0x0000000000000072 <+34>:	mov    rsi,QWORD PTR [rdi+0x10]
   0x00000000000000bb <+107>:	movss  xmm5,DWORD PTR [rsi+rdi*1]
   0x00000000000000c0 <+112>:	divss  xmm5,xmm2
   0x00000000000000c4 <+116>:	movss  DWORD PTR [r8+rdi*1],xmm5
   0x00000000000000ed <+157>:	movss  xmm8,DWORD PTR [rsi+rdi*1+0x4]
   0x00000000000000f4 <+164>:	divss  xmm8,xmm2
   0x00000000000000f9 <+169>:	movss  DWORD PTR [r8+rdi*1+0x4],xmm8
   0x0000000000000125 <+213>:	movss  xmm11,DWORD PTR [rsi+rdi*1+0x8]
   0x000000000000012c <+220>:	divss  xmm11,xmm2
   0x0000000000000131 <+225>:	movss  DWORD PTR [r8+rdi*1+0x8],xmm11
   0x000000000000015d <+269>:	movss  xmm14,DWORD PTR [rsi+rdi*1+0xc]
   0x0000000000000164 <+276>:	divss  xmm14,xmm2
   0x0000000000000169 <+281>:	movss  DWORD PTR [r8+rdi*1+0xc],xmm14
   0x0000000000000191 <+321>:	movss  xmm4,DWORD PTR [rsi+rdi*1+0x10]
   0x0000000000000197 <+327>:	divss  xmm4,xmm2
   0x000000000000019b <+331>:	movss  DWORD PTR [r8+rdi*1+0x10],xmm4
   0x00000000000001c2 <+370>:	movss  xmm7,DWORD PTR [rsi+rdi*1+0x14]
   0x00000000000001c8 <+376>:	divss  xmm7,xmm2
   0x00000000000001cc <+380>:	movss  DWORD PTR [r8+rdi*1+0x14],xmm7
   0x00000000000001f4 <+420>:	movss  xmm10,DWORD PTR [rsi+rdi*1+0x18]
   0x00000000000001fb <+427>:	divss  xmm10,xmm2
   0x0000000000000200 <+432>:	movss  DWORD PTR [r8+rdi*1+0x18],xmm10
   0x0000000000000228 <+472>:	movss  xmm13,DWORD PTR [rsi+rdi*1+0x1c]
   0x000000000000022f <+479>:	divss  xmm13,xmm2
   0x0000000000000234 <+484>:	movss  DWORD PTR [r8+rdi*1+0x1c],xmm13
   0x0000000000000288 <+568>:	movss  xmm5,DWORD PTR [rsi+rdi*4+0x14]
   0x000000000000028e <+574>:	divss  xmm5,xmm2
   0x0000000000000292 <+578>:	movss  DWORD PTR [r8+rdi*4+0x14],xmm5
   0x00000000000002bc <+620>:	movss  xmm5,DWORD PTR [rsi+rdi*4+0x10]
   0x00000000000002c2 <+626>:	divss  xmm5,xmm2
   0x00000000000002c6 <+630>:	movss  DWORD PTR [r8+rdi*4+0x10],xmm5
   0x00000000000002f0 <+672>:	movss  xmm5,DWORD PTR [rsi+rdi*4+0xc]
   0x00000000000002f6 <+678>:	divss  xmm5,xmm2
   0x00000000000002fa <+682>:	movss  DWORD PTR [r8+rdi*4+0xc],xmm5
   0x0000000000000324 <+724>:	movss  xmm5,DWORD PTR [rsi+rdi*4+0x8]
   0x000000000000032a <+730>:	divss  xmm5,xmm2
   0x000000000000032e <+734>:	movss  DWORD PTR [r8+rdi*4+0x8],xmm5
   0x0000000000000358 <+776>:	movss  xmm5,DWORD PTR [rsi+rdi*4+0x4]
   0x000000000000035e <+782>:	divss  xmm5,xmm2
   0x0000000000000362 <+786>:	movss  DWORD PTR [r8+rdi*4+0x4],xmm5
   0x0000000000000388 <+824>:	movss  xmm5,DWORD PTR [rsi+rdi*4]
   0x000000000000038d <+829>:	divss  xmm5,xmm2
   0x0000000000000391 <+833>:	movss  DWORD PTR [r8+rdi*4],xmm5
   0x00000000000003b7 <+871>:	movss  xmm0,DWORD PTR [rsi+rdi*4-0x4]
   0x00000000000003bd <+877>:	divss  xmm0,xmm2
   0x00000000000003c1 <+881>:	movss  DWORD PTR [r8+rdi*4-0x4],xmm0

47	  }
48	}
   0x00000000000003c8 <+888>:	pop    r14
   0x00000000000003ca <+890>:	ret    
   0x00000000000003cb <+891>:	nop    DWORD PTR [rax+rax*1+0x0]

End of assembler dump.
