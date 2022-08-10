for function floating_division_aos:
b, float c, struct pixel** arr, struct pixel** results, size_t arr_len) {


__INTEL_COMPILER)

are specific for the compiler and platform in use. So the best bet is to look at compiler's documentation.


ivdep

i = 0; i < arr_len; ++i) {
<+0>:	xor    eax,eax
<+2>:	test   rdx,rdx
<+5>:	jbe    0x41 <floating_division_aos+65>
<+18>:	inc    rax
<+60>:	cmp    rax,rdx
<+63>:	jb     0x7 <floating_division_aos+7>

results[i]->r = arr[i]->r / a;
<+7>:	mov    rcx,QWORD PTR [rdi+rax*8]
<+14>:	mov    r8,QWORD PTR [rsi+rax*8]
<+21>:	movss  xmm3,DWORD PTR [rcx]
<+35>:	divss  xmm3,xmm0
<+43>:	movss  DWORD PTR [r8],xmm3

results[i]->g = b / arr[i]->g;
<+11>:	movaps xmm4,xmm1
<+30>:	divss  xmm4,DWORD PTR [rcx+0x4]
<+48>:	movss  DWORD PTR [r8+0x4],xmm4

results[i]->b = arr[i]->b / c;
<+25>:	movss  xmm5,DWORD PTR [rcx+0x8]
<+39>:	divss  xmm5,xmm2
<+54>:	movss  DWORD PTR [r8+0x8],xmm5


35	}
<+65>:	ret    
<+66>:	nop    DWORD PTR [rax+0x0]
<+73>:	nop    DWORD PTR [rax+0x0]


for function floating_division_soa:
b, float c, struct pixelArray* arr, struct pixelArray* results, size_t arr_len) {
<+0>:	push   r14
<+11>:	mov    rax,rdx
<+43>:	shr    rax,0x3
<+518>:	sub    rdx,rdi
<+521>:	inc    rdx
<+524>:	dec    edx

__INTEL_COMPILER)


ivdep

i = 0; i < arr_len; ++i) {
<+2>:	test   rdx,rdx
<+5>:	jbe    0x3c8 <floating_division_soa+888>
<+14>:	xor    ecx,ecx
<+38>:	mov    edi,0x1
<+47>:	je     0x24d <floating_division_soa+509>
<+74>:	inc    ecx
<+491>:	cmp    rcx,rax
<+494>:	jb     0x85 <floating_division_soa+53>
<+500>:	shl    ecx,0x3
<+503>:	movsxd rdi,ecx
<+506>:	inc    rdi
<+509>:	cmp    rdi,rdx
<+512>:	ja     0x3c8 <floating_division_soa+888>
<+526>:	jmp    QWORD PTR [rdx*8+0x0]

results->r[i] = arr->r[i] / a;
<+16>:	mov    r10,QWORD PTR [rsi]
<+27>:	mov    r11,QWORD PTR [rdi]
<+53>:	movsxd rdi,ecx
<+59>:	shl    rdi,0x5
<+76>:	movss  xmm3,DWORD PTR [r11+rdi*1]
<+82>:	divss  xmm3,xmm0
<+86>:	movss  DWORD PTR [r10+rdi*1],xmm3
<+122>:	movss  xmm6,DWORD PTR [r11+rdi*1+0x4]
<+129>:	divss  xmm6,xmm0
<+133>:	movss  DWORD PTR [r10+rdi*1+0x4],xmm6
<+176>:	movss  xmm9,DWORD PTR [r11+rdi*1+0x8]
<+183>:	divss  xmm9,xmm0
<+188>:	movss  DWORD PTR [r10+rdi*1+0x8],xmm9
<+232>:	movss  xmm12,DWORD PTR [r11+rdi*1+0xc]
<+239>:	divss  xmm12,xmm0
<+244>:	movss  DWORD PTR [r10+rdi*1+0xc],xmm12
<+288>:	movss  xmm15,DWORD PTR [r11+rdi*1+0x10]
<+295>:	divss  xmm15,xmm0
<+300>:	movss  DWORD PTR [r10+rdi*1+0x10],xmm15
<+338>:	movss  xmm5,DWORD PTR [r11+rdi*1+0x14]
<+345>:	divss  xmm5,xmm0
<+349>:	movss  DWORD PTR [r10+rdi*1+0x14],xmm5
<+387>:	movss  xmm8,DWORD PTR [r11+rdi*1+0x18]
<+394>:	divss  xmm8,xmm0
<+399>:	movss  DWORD PTR [r10+rdi*1+0x18],xmm8
<+439>:	movss  xmm11,DWORD PTR [r11+rdi*1+0x1c]
<+446>:	divss  xmm11,xmm0
<+451>:	movss  DWORD PTR [r10+rdi*1+0x1c],xmm11
<+533>:	movss  xmm3,DWORD PTR [r11+rdi*4+0x14]
<+543>:	divss  xmm3,xmm0
<+547>:	movss  DWORD PTR [r10+rdi*4+0x14],xmm3
<+585>:	movss  xmm3,DWORD PTR [r11+rdi*4+0x10]
<+595>:	divss  xmm3,xmm0
<+599>:	movss  DWORD PTR [r10+rdi*4+0x10],xmm3
<+637>:	movss  xmm3,DWORD PTR [r11+rdi*4+0xc]
<+647>:	divss  xmm3,xmm0
<+651>:	movss  DWORD PTR [r10+rdi*4+0xc],xmm3
<+689>:	movss  xmm3,DWORD PTR [r11+rdi*4+0x8]
<+699>:	divss  xmm3,xmm0
<+703>:	movss  DWORD PTR [r10+rdi*4+0x8],xmm3
<+741>:	movss  xmm3,DWORD PTR [r11+rdi*4+0x4]
<+751>:	divss  xmm3,xmm0
<+755>:	movss  DWORD PTR [r10+rdi*4+0x4],xmm3
<+793>:	movss  xmm3,DWORD PTR [r11+rdi*4]
<+802>:	divss  xmm3,xmm0
<+806>:	movss  DWORD PTR [r10+rdi*4],xmm3
<+839>:	movss  xmm3,DWORD PTR [r11+rdi*4-0x4]
<+846>:	divss  xmm3,xmm0
<+850>:	movss  DWORD PTR [r10+rdi*4-0x4],xmm3

results->g[i] = b / arr->g[i];
<+19>:	mov    r14,QWORD PTR [rsi+0x8]
<+30>:	mov    r9,QWORD PTR [rdi+0x8]
<+56>:	movaps xmm4,xmm1
<+63>:	movaps xmm7,xmm1
<+66>:	movaps xmm10,xmm1
<+70>:	movaps xmm13,xmm1
<+92>:	movaps xmm3,xmm1
<+95>:	divss  xmm4,DWORD PTR [r9+rdi*1]
<+101>:	movss  DWORD PTR [r14+rdi*1],xmm4
<+140>:	movaps xmm6,xmm1
<+143>:	divss  xmm7,DWORD PTR [r9+rdi*1+0x4]
<+150>:	movss  DWORD PTR [r14+rdi*1+0x4],xmm7
<+195>:	movaps xmm9,xmm1
<+199>:	divss  xmm10,DWORD PTR [r9+rdi*1+0x8]
<+206>:	movss  DWORD PTR [r14+rdi*1+0x8],xmm10
<+251>:	movaps xmm12,xmm1
<+255>:	divss  xmm13,DWORD PTR [r9+rdi*1+0xc]
<+262>:	movss  DWORD PTR [r14+rdi*1+0xc],xmm13
<+307>:	divss  xmm3,DWORD PTR [r9+rdi*1+0x10]
<+314>:	movss  DWORD PTR [r14+rdi*1+0x10],xmm3
<+356>:	divss  xmm6,DWORD PTR [r9+rdi*1+0x14]
<+363>:	movss  DWORD PTR [r14+rdi*1+0x14],xmm6
<+406>:	divss  xmm9,DWORD PTR [r9+rdi*1+0x18]
<+413>:	movss  DWORD PTR [r14+rdi*1+0x18],xmm9
<+458>:	divss  xmm12,DWORD PTR [r9+rdi*1+0x1c]
<+465>:	movss  DWORD PTR [r14+rdi*1+0x1c],xmm12
<+540>:	movaps xmm4,xmm1
<+554>:	divss  xmm4,DWORD PTR [r9+rdi*4+0x14]
<+561>:	movss  DWORD PTR [r14+rdi*4+0x14],xmm4
<+592>:	movaps xmm4,xmm1
<+606>:	divss  xmm4,DWORD PTR [r9+rdi*4+0x10]
<+613>:	movss  DWORD PTR [r14+rdi*4+0x10],xmm4
<+644>:	movaps xmm4,xmm1
<+658>:	divss  xmm4,DWORD PTR [r9+rdi*4+0xc]
<+665>:	movss  DWORD PTR [r14+rdi*4+0xc],xmm4
<+696>:	movaps xmm4,xmm1
<+710>:	divss  xmm4,DWORD PTR [r9+rdi*4+0x8]
<+717>:	movss  DWORD PTR [r14+rdi*4+0x8],xmm4
<+748>:	movaps xmm4,xmm1
<+762>:	divss  xmm4,DWORD PTR [r9+rdi*4+0x4]
<+769>:	movss  DWORD PTR [r14+rdi*4+0x4],xmm4
<+799>:	movaps xmm4,xmm1
<+812>:	divss  xmm4,DWORD PTR [r9+rdi*4]
<+818>:	movss  DWORD PTR [r14+rdi*4],xmm4
<+857>:	divss  xmm1,DWORD PTR [r9+rdi*4-0x4]
<+864>:	movss  DWORD PTR [r14+rdi*4-0x4],xmm1

results->b[i] = arr->b[i] / c;
<+23>:	mov    r8,QWORD PTR [rsi+0x10]
<+34>:	mov    rsi,QWORD PTR [rdi+0x10]
<+107>:	movss  xmm5,DWORD PTR [rsi+rdi*1]
<+112>:	divss  xmm5,xmm2
<+116>:	movss  DWORD PTR [r8+rdi*1],xmm5
<+157>:	movss  xmm8,DWORD PTR [rsi+rdi*1+0x4]
<+164>:	divss  xmm8,xmm2
<+169>:	movss  DWORD PTR [r8+rdi*1+0x4],xmm8
<+213>:	movss  xmm11,DWORD PTR [rsi+rdi*1+0x8]
<+220>:	divss  xmm11,xmm2
<+225>:	movss  DWORD PTR [r8+rdi*1+0x8],xmm11
<+269>:	movss  xmm14,DWORD PTR [rsi+rdi*1+0xc]
<+276>:	divss  xmm14,xmm2
<+281>:	movss  DWORD PTR [r8+rdi*1+0xc],xmm14
<+321>:	movss  xmm4,DWORD PTR [rsi+rdi*1+0x10]
<+327>:	divss  xmm4,xmm2
<+331>:	movss  DWORD PTR [r8+rdi*1+0x10],xmm4
<+370>:	movss  xmm7,DWORD PTR [rsi+rdi*1+0x14]
<+376>:	divss  xmm7,xmm2
<+380>:	movss  DWORD PTR [r8+rdi*1+0x14],xmm7
<+420>:	movss  xmm10,DWORD PTR [rsi+rdi*1+0x18]
<+427>:	divss  xmm10,xmm2
<+432>:	movss  DWORD PTR [r8+rdi*1+0x18],xmm10
<+472>:	movss  xmm13,DWORD PTR [rsi+rdi*1+0x1c]
<+479>:	divss  xmm13,xmm2
<+484>:	movss  DWORD PTR [r8+rdi*1+0x1c],xmm13
<+568>:	movss  xmm5,DWORD PTR [rsi+rdi*4+0x14]
<+574>:	divss  xmm5,xmm2
<+578>:	movss  DWORD PTR [r8+rdi*4+0x14],xmm5
<+620>:	movss  xmm5,DWORD PTR [rsi+rdi*4+0x10]
<+626>:	divss  xmm5,xmm2
<+630>:	movss  DWORD PTR [r8+rdi*4+0x10],xmm5
<+672>:	movss  xmm5,DWORD PTR [rsi+rdi*4+0xc]
<+678>:	divss  xmm5,xmm2
<+682>:	movss  DWORD PTR [r8+rdi*4+0xc],xmm5
<+724>:	movss  xmm5,DWORD PTR [rsi+rdi*4+0x8]
<+730>:	divss  xmm5,xmm2
<+734>:	movss  DWORD PTR [r8+rdi*4+0x8],xmm5
<+776>:	movss  xmm5,DWORD PTR [rsi+rdi*4+0x4]
<+782>:	divss  xmm5,xmm2
<+786>:	movss  DWORD PTR [r8+rdi*4+0x4],xmm5
<+824>:	movss  xmm5,DWORD PTR [rsi+rdi*4]
<+829>:	divss  xmm5,xmm2
<+833>:	movss  DWORD PTR [r8+rdi*4],xmm5
<+871>:	movss  xmm0,DWORD PTR [rsi+rdi*4-0x4]
<+877>:	divss  xmm0,xmm2
<+881>:	movss  DWORD PTR [r8+rdi*4-0x4],xmm0


48	}
<+888>:	pop    r14
<+890>:	ret    
<+891>:	nop    DWORD PTR [rax+rax*1+0x0]


