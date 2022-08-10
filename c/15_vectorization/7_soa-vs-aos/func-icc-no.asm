for function floating_division_aos:
b, float c, struct pixel** arr, struct pixel** results, size_t arr_len) {




are specific for the compiler and platform in use. So the best bet is to look at compiler's documentation.


i = 0; i < arr_len; ++i) {
<+0>:	xor    %eax,%eax
<+2>:	test   %rdx,%rdx
<+5>:	jbe    0x41 <floating_division_aos+65>
<+18>:	inc    %rax
<+60>:	cmp    %rdx,%rax
<+63>:	jb     0x7 <floating_division_aos+7>

results[i]->r = arr[i]->r / a;
<+7>:	mov    (%rdi,%rax,8),%rcx
<+14>:	mov    (%rsi,%rax,8),%r8
<+21>:	movss  (%rcx),%xmm3
<+35>:	divss  %xmm0,%xmm3
<+43>:	movss  %xmm3,(%r8)

results[i]->g = b / arr[i]->g;
<+11>:	movaps %xmm1,%xmm4
<+30>:	divss  0x4(%rcx),%xmm4
<+48>:	movss  %xmm4,0x4(%r8)

results[i]->b = arr[i]->b / c;
<+25>:	movss  0x8(%rcx),%xmm5
<+39>:	divss  %xmm2,%xmm5
<+54>:	movss  %xmm5,0x8(%r8)


33	}
<+65>:	ret    
<+66>:	nopl   0x0(%rax)
<+73>:	nopl   0x0(%rax)


for function floating_division_soa:
b, float c, struct pixelArray* arr, struct pixelArray* results, size_t arr_len) {
<+0>:	push   %r14
<+11>:	mov    %rdx,%rax
<+43>:	shr    $0x3,%rax
<+518>:	sub    %rdi,%rdx
<+521>:	inc    %rdx
<+524>:	dec    %edx


 

i = 0; i < arr_len; ++i) {
<+2>:	test   %rdx,%rdx
<+5>:	jbe    0x3c8 <floating_division_soa+888>
<+14>:	xor    %ecx,%ecx
<+38>:	mov    $0x1,%edi
<+47>:	je     0x24d <floating_division_soa+509>
<+74>:	inc    %ecx
<+491>:	cmp    %rax,%rcx
<+494>:	jb     0x85 <floating_division_soa+53>
<+500>:	shl    $0x3,%ecx
<+503>:	movslq %ecx,%rdi
<+506>:	inc    %rdi
<+509>:	cmp    %rdx,%rdi
<+512>:	ja     0x3c8 <floating_division_soa+888>
<+526>:	jmp    *0x0(,%rdx,8)

results->r[i] = arr->r[i] / a;
<+16>:	mov    (%rsi),%r10
<+27>:	mov    (%rdi),%r11
<+53>:	movslq %ecx,%rdi
<+59>:	shl    $0x5,%rdi
<+76>:	movss  (%r11,%rdi,1),%xmm3
<+82>:	divss  %xmm0,%xmm3
<+86>:	movss  %xmm3,(%r10,%rdi,1)
<+122>:	movss  0x4(%r11,%rdi,1),%xmm6
<+129>:	divss  %xmm0,%xmm6
<+133>:	movss  %xmm6,0x4(%r10,%rdi,1)
<+176>:	movss  0x8(%r11,%rdi,1),%xmm9
<+183>:	divss  %xmm0,%xmm9
<+188>:	movss  %xmm9,0x8(%r10,%rdi,1)
<+232>:	movss  0xc(%r11,%rdi,1),%xmm12
<+239>:	divss  %xmm0,%xmm12
<+244>:	movss  %xmm12,0xc(%r10,%rdi,1)
<+288>:	movss  0x10(%r11,%rdi,1),%xmm15
<+295>:	divss  %xmm0,%xmm15
<+300>:	movss  %xmm15,0x10(%r10,%rdi,1)
<+338>:	movss  0x14(%r11,%rdi,1),%xmm5
<+345>:	divss  %xmm0,%xmm5
<+349>:	movss  %xmm5,0x14(%r10,%rdi,1)
<+387>:	movss  0x18(%r11,%rdi,1),%xmm8
<+394>:	divss  %xmm0,%xmm8
<+399>:	movss  %xmm8,0x18(%r10,%rdi,1)
<+439>:	movss  0x1c(%r11,%rdi,1),%xmm11
<+446>:	divss  %xmm0,%xmm11
<+451>:	movss  %xmm11,0x1c(%r10,%rdi,1)
<+533>:	movss  0x14(%r11,%rdi,4),%xmm3
<+543>:	divss  %xmm0,%xmm3
<+547>:	movss  %xmm3,0x14(%r10,%rdi,4)
<+585>:	movss  0x10(%r11,%rdi,4),%xmm3
<+595>:	divss  %xmm0,%xmm3
<+599>:	movss  %xmm3,0x10(%r10,%rdi,4)
<+637>:	movss  0xc(%r11,%rdi,4),%xmm3
<+647>:	divss  %xmm0,%xmm3
<+651>:	movss  %xmm3,0xc(%r10,%rdi,4)
<+689>:	movss  0x8(%r11,%rdi,4),%xmm3
<+699>:	divss  %xmm0,%xmm3
<+703>:	movss  %xmm3,0x8(%r10,%rdi,4)
<+741>:	movss  0x4(%r11,%rdi,4),%xmm3
<+751>:	divss  %xmm0,%xmm3
<+755>:	movss  %xmm3,0x4(%r10,%rdi,4)
<+793>:	movss  (%r11,%rdi,4),%xmm3
<+802>:	divss  %xmm0,%xmm3
<+806>:	movss  %xmm3,(%r10,%rdi,4)
<+839>:	movss  -0x4(%r11,%rdi,4),%xmm3
<+846>:	divss  %xmm0,%xmm3
<+850>:	movss  %xmm3,-0x4(%r10,%rdi,4)

results->g[i] = b / arr->g[i];
<+19>:	mov    0x8(%rsi),%r14
<+30>:	mov    0x8(%rdi),%r9
<+56>:	movaps %xmm1,%xmm4
<+63>:	movaps %xmm1,%xmm7
<+66>:	movaps %xmm1,%xmm10
<+70>:	movaps %xmm1,%xmm13
<+92>:	movaps %xmm1,%xmm3
<+95>:	divss  (%r9,%rdi,1),%xmm4
<+101>:	movss  %xmm4,(%r14,%rdi,1)
<+140>:	movaps %xmm1,%xmm6
<+143>:	divss  0x4(%r9,%rdi,1),%xmm7
<+150>:	movss  %xmm7,0x4(%r14,%rdi,1)
<+195>:	movaps %xmm1,%xmm9
<+199>:	divss  0x8(%r9,%rdi,1),%xmm10
<+206>:	movss  %xmm10,0x8(%r14,%rdi,1)
<+251>:	movaps %xmm1,%xmm12
<+255>:	divss  0xc(%r9,%rdi,1),%xmm13
<+262>:	movss  %xmm13,0xc(%r14,%rdi,1)
<+307>:	divss  0x10(%r9,%rdi,1),%xmm3
<+314>:	movss  %xmm3,0x10(%r14,%rdi,1)
<+356>:	divss  0x14(%r9,%rdi,1),%xmm6
<+363>:	movss  %xmm6,0x14(%r14,%rdi,1)
<+406>:	divss  0x18(%r9,%rdi,1),%xmm9
<+413>:	movss  %xmm9,0x18(%r14,%rdi,1)
<+458>:	divss  0x1c(%r9,%rdi,1),%xmm12
<+465>:	movss  %xmm12,0x1c(%r14,%rdi,1)
<+540>:	movaps %xmm1,%xmm4
<+554>:	divss  0x14(%r9,%rdi,4),%xmm4
<+561>:	movss  %xmm4,0x14(%r14,%rdi,4)
<+592>:	movaps %xmm1,%xmm4
<+606>:	divss  0x10(%r9,%rdi,4),%xmm4
<+613>:	movss  %xmm4,0x10(%r14,%rdi,4)
<+644>:	movaps %xmm1,%xmm4
<+658>:	divss  0xc(%r9,%rdi,4),%xmm4
<+665>:	movss  %xmm4,0xc(%r14,%rdi,4)
<+696>:	movaps %xmm1,%xmm4
<+710>:	divss  0x8(%r9,%rdi,4),%xmm4
<+717>:	movss  %xmm4,0x8(%r14,%rdi,4)
<+748>:	movaps %xmm1,%xmm4
<+762>:	divss  0x4(%r9,%rdi,4),%xmm4
<+769>:	movss  %xmm4,0x4(%r14,%rdi,4)
<+799>:	movaps %xmm1,%xmm4
<+812>:	divss  (%r9,%rdi,4),%xmm4
<+818>:	movss  %xmm4,(%r14,%rdi,4)
<+857>:	divss  -0x4(%r9,%rdi,4),%xmm1
<+864>:	movss  %xmm1,-0x4(%r14,%rdi,4)

results->b[i] = arr->b[i] / c;
<+23>:	mov    0x10(%rsi),%r8
<+34>:	mov    0x10(%rdi),%rsi
<+107>:	movss  (%rsi,%rdi,1),%xmm5
<+112>:	divss  %xmm2,%xmm5
<+116>:	movss  %xmm5,(%r8,%rdi,1)
<+157>:	movss  0x4(%rsi,%rdi,1),%xmm8
<+164>:	divss  %xmm2,%xmm8
<+169>:	movss  %xmm8,0x4(%r8,%rdi,1)
<+213>:	movss  0x8(%rsi,%rdi,1),%xmm11
<+220>:	divss  %xmm2,%xmm11
<+225>:	movss  %xmm11,0x8(%r8,%rdi,1)
<+269>:	movss  0xc(%rsi,%rdi,1),%xmm14
<+276>:	divss  %xmm2,%xmm14
<+281>:	movss  %xmm14,0xc(%r8,%rdi,1)
<+321>:	movss  0x10(%rsi,%rdi,1),%xmm4
<+327>:	divss  %xmm2,%xmm4
<+331>:	movss  %xmm4,0x10(%r8,%rdi,1)
<+370>:	movss  0x14(%rsi,%rdi,1),%xmm7
<+376>:	divss  %xmm2,%xmm7
<+380>:	movss  %xmm7,0x14(%r8,%rdi,1)
<+420>:	movss  0x18(%rsi,%rdi,1),%xmm10
<+427>:	divss  %xmm2,%xmm10
<+432>:	movss  %xmm10,0x18(%r8,%rdi,1)
<+472>:	movss  0x1c(%rsi,%rdi,1),%xmm13
<+479>:	divss  %xmm2,%xmm13
<+484>:	movss  %xmm13,0x1c(%r8,%rdi,1)
<+568>:	movss  0x14(%rsi,%rdi,4),%xmm5
<+574>:	divss  %xmm2,%xmm5
<+578>:	movss  %xmm5,0x14(%r8,%rdi,4)
<+620>:	movss  0x10(%rsi,%rdi,4),%xmm5
<+626>:	divss  %xmm2,%xmm5
<+630>:	movss  %xmm5,0x10(%r8,%rdi,4)
<+672>:	movss  0xc(%rsi,%rdi,4),%xmm5
<+678>:	divss  %xmm2,%xmm5
<+682>:	movss  %xmm5,0xc(%r8,%rdi,4)
<+724>:	movss  0x8(%rsi,%rdi,4),%xmm5
<+730>:	divss  %xmm2,%xmm5
<+734>:	movss  %xmm5,0x8(%r8,%rdi,4)
<+776>:	movss  0x4(%rsi,%rdi,4),%xmm5
<+782>:	divss  %xmm2,%xmm5
<+786>:	movss  %xmm5,0x4(%r8,%rdi,4)
<+824>:	movss  (%rsi,%rdi,4),%xmm5
<+829>:	divss  %xmm2,%xmm5
<+833>:	movss  %xmm5,(%r8,%rdi,4)
<+871>:	movss  -0x4(%rsi,%rdi,4),%xmm0
<+877>:	divss  %xmm2,%xmm0
<+881>:	movss  %xmm0,-0x4(%r8,%rdi,4)


44	}
<+888>:	pop    %r14
<+890>:	ret    
<+891>:	nopl   0x0(%rax,%rax,1)


