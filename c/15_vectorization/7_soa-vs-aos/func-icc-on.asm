for function floating_division_aos:
b, float c, struct pixel** arr, struct pixel** results, size_t arr_len) {
<+0>:	mov    %rsi,%r8
<+3>:	movaps %xmm2,%xmm5
<+6>:	movaps %xmm1,%xmm4
<+9>:	movaps %xmm0,%xmm3
<+180>:	movaps %xmm3,%xmm0
<+183>:	movaps %xmm5,%xmm6
<+186>:	shufps $0x0,%xmm0,%xmm0
<+190>:	movaps %xmm4,%xmm2
<+205>:	shufps $0x0,%xmm6,%xmm6
<+224>:	shufps $0x0,%xmm2,%xmm2
<+666>:	movaps %xmm3,%xmm0
<+669>:	movaps %xmm5,%xmm6
<+672>:	shufps $0x0,%xmm0,%xmm0
<+676>:	movaps %xmm4,%xmm2
<+691>:	shufps $0x0,%xmm6,%xmm6
<+710>:	shufps $0x0,%xmm2,%xmm2




are specific for the compiler and platform in use. So the best bet is to look at compiler's documentation.


i = 0; i < arr_len; ++i) {
<+12>:	test   %rdx,%rdx
<+15>:	jbe    0x4c1 <floating_division_aos+1217>
<+21>:	cmp    $0x4,%rdx
<+25>:	jb     0x4c2 <floating_division_aos+1218>
<+31>:	mov    %rdi,%rcx
<+34>:	and    $0xf,%rcx
<+38>:	je     0x3a <floating_division_aos+58>
<+40>:	test   $0x7,%rcx
<+47>:	jne    0x4c2 <floating_division_aos+1218>
<+53>:	mov    $0x1,%ecx
<+58>:	lea    0x4(%rcx),%rax
<+62>:	cmp    %rax,%rdx
<+65>:	jb     0x4c2 <floating_division_aos+1218>
<+71>:	mov    %rdx,%rsi
<+74>:	xor    %r9d,%r9d
<+77>:	sub    %rcx,%rsi
<+80>:	xor    %eax,%eax
<+82>:	and    $0x3,%rsi
<+86>:	neg    %rsi
<+89>:	add    %rdx,%rsi
<+92>:	test   %rcx,%rcx
<+95>:	jbe    0xa1 <floating_division_aos+161>
<+108>:	inc    %r9d
<+111>:	inc    %rax
<+156>:	cmp    %rcx,%r9
<+159>:	jb     0x61 <floating_division_aos+97>
<+167>:	test   $0xf,%r9
<+174>:	je     0x29a <floating_division_aos+666>
<+235>:	add    $0x4,%eax
<+624>:	add    $0x4,%rcx
<+652>:	cmp    %rsi,%rax
<+655>:	jb     0xe7 <floating_division_aos+231>
<+661>:	jmp    0x479 <floating_division_aos+1145>
<+721>:	add    $0x4,%eax
<+1108>:	add    $0x4,%rcx
<+1136>:	cmp    %rsi,%rax
<+1139>:	jb     0x2cd <floating_division_aos+717>
<+1145>:	movslq %esi,%rax
<+1148>:	mov    %esi,%ecx
<+1150>:	mov    %esi,%esi
<+1152>:	cmp    %rdx,%rsi
<+1155>:	jae    0x4c1 <floating_division_aos+1217>
<+1168>:	inc    %ecx
<+1170>:	inc    %rax
<+1212>:	cmp    %rdx,%rcx
<+1215>:	jb     0x485 <floating_division_aos+1157>
<+1218>:	xor    %esi,%esi
<+1220>:	jmp    0x479 <floating_division_aos+1145>
<+1222>:	nopl   (%rax)
<+1225>:	nopl   0x0(%rax)

results[i]->r = arr[i]->r / a;
<+97>:	mov    (%rdi,%rax,8),%r10
<+104>:	mov    (%r8,%rax,8),%r11
<+114>:	movss  (%r10),%xmm0
<+131>:	divss  %xmm3,%xmm0
<+139>:	movss  %xmm0,(%r11)
<+161>:	mov    %ecx,%eax
<+193>:	rcpps  %xmm0,%xmm1
<+196>:	mulps  %xmm1,%xmm0
<+199>:	mulps  %xmm1,%xmm0
<+202>:	addps  %xmm1,%xmm1
<+209>:	subps  %xmm0,%xmm1
<+231>:	mov    (%rdi,%rcx,8),%r9
<+238>:	mov    0x8(%rdi,%rcx,8),%r10
<+243>:	mov    0x10(%rdi,%rcx,8),%r11
<+248>:	movd   (%r9),%xmm9
<+253>:	mov    0x18(%rdi,%rcx,8),%r9
<+258>:	movd   (%r10),%xmm6
<+263>:	movd   (%r11),%xmm8
<+268>:	movd   (%r9),%xmm7
<+273>:	punpckldq %xmm6,%xmm9
<+278>:	punpckldq %xmm7,%xmm8
<+283>:	movlhps %xmm8,%xmm9
<+287>:	mulps  %xmm1,%xmm9
<+291>:	movdqu 0x10(%r8,%rcx,8),%xmm11
<+298>:	mov    (%r8,%rcx,8),%r10
<+302>:	movaps %xmm9,%xmm12
<+306>:	movq   %xmm11,%r9
<+311>:	punpckhqdq %xmm11,%xmm11
<+316>:	movd   %xmm9,(%r10)
<+321>:	movq   %xmm11,%r10
<+326>:	movhlps %xmm9,%xmm12
<+330>:	pshuflw $0xee,%xmm9,%xmm10
<+336>:	mov    0x8(%r8,%rcx,8),%r11
<+341>:	pshuflw $0xee,%xmm12,%xmm13
<+347>:	movd   %xmm10,(%r11)
<+352>:	movd   %xmm12,(%r9)
<+357>:	movd   %xmm13,(%r10)
<+679>:	rcpps  %xmm0,%xmm1
<+682>:	mulps  %xmm1,%xmm0
<+685>:	mulps  %xmm1,%xmm0
<+688>:	addps  %xmm1,%xmm1
<+695>:	subps  %xmm0,%xmm1
<+717>:	mov    (%rdi,%rcx,8),%r9
<+724>:	mov    0x8(%rdi,%rcx,8),%r10
<+729>:	mov    0x10(%rdi,%rcx,8),%r11
<+734>:	movd   (%r9),%xmm9
<+739>:	mov    0x18(%rdi,%rcx,8),%r9
<+744>:	movd   (%r10),%xmm6
<+749>:	movd   (%r11),%xmm8
<+754>:	movd   (%r9),%xmm7
<+759>:	punpckldq %xmm6,%xmm9
<+764>:	punpckldq %xmm7,%xmm8
<+769>:	movlhps %xmm8,%xmm9
<+773>:	mulps  %xmm1,%xmm9
<+777>:	mov    (%r8,%rcx,8),%r10
<+781>:	movaps %xmm9,%xmm12
<+785>:	movdqu 0x10(%r8,%rcx,8),%xmm11
<+792>:	movq   %xmm11,%r9
<+797>:	punpckhqdq %xmm11,%xmm11
<+802>:	movd   %xmm9,(%r10)
<+807>:	movq   %xmm11,%r10
<+812>:	movhlps %xmm9,%xmm12
<+816>:	pshuflw $0xee,%xmm9,%xmm10
<+822>:	mov    0x8(%r8,%rcx,8),%r11
<+827>:	pshuflw $0xee,%xmm12,%xmm13
<+833>:	movd   %xmm10,(%r11)
<+838>:	movd   %xmm12,(%r9)
<+843>:	movd   %xmm13,(%r10)
<+1157>:	mov    (%rdi,%rax,8),%rsi
<+1164>:	mov    (%r8,%rax,8),%r9
<+1173>:	movss  (%rsi),%xmm0
<+1187>:	divss  %xmm3,%xmm0
<+1195>:	movss  %xmm0,(%r9)

results[i]->g = b / arr[i]->g;
<+101>:	movaps %xmm4,%xmm1
<+125>:	divss  0x4(%r10),%xmm1
<+144>:	movss  %xmm1,0x4(%r11)
<+362>:	mov    (%rdi,%rcx,8),%r11
<+366>:	mov    0x8(%rdi,%rcx,8),%r9
<+371>:	mov    0x10(%rdi,%rcx,8),%r10
<+376>:	movd   0x4(%r11),%xmm6
<+382>:	mov    0x18(%rdi,%rcx,8),%r11
<+387>:	movd   0x4(%r9),%xmm14
<+393>:	punpckldq %xmm14,%xmm6
<+398>:	movd   0x4(%r10),%xmm14
<+404>:	movd   0x4(%r11),%xmm15
<+410>:	punpckldq %xmm15,%xmm14
<+415>:	movlhps %xmm14,%xmm6
<+419>:	rcpps  %xmm6,%xmm7
<+422>:	movdqu 0x10(%r8,%rcx,8),%xmm9
<+429>:	mulps  %xmm7,%xmm6
<+432>:	movq   %xmm9,%r11
<+437>:	punpckhqdq %xmm9,%xmm9
<+442>:	mulps  %xmm7,%xmm6
<+445>:	addps  %xmm7,%xmm7
<+448>:	mov    (%r8,%rcx,8),%r9
<+452>:	subps  %xmm6,%xmm7
<+455>:	mulps  %xmm2,%xmm7
<+458>:	movd   %xmm7,0x4(%r9)
<+464>:	movaps %xmm7,%xmm10
<+468>:	movq   %xmm9,%r9
<+473>:	movhlps %xmm7,%xmm10
<+477>:	pshuflw $0xee,%xmm7,%xmm8
<+483>:	mov    0x8(%r8,%rcx,8),%r10
<+488>:	pshuflw $0xee,%xmm10,%xmm11
<+494>:	movd   %xmm8,0x4(%r10)
<+500>:	movd   %xmm10,0x4(%r11)
<+506>:	movd   %xmm11,0x4(%r9)
<+848>:	mov    (%rdi,%rcx,8),%r11
<+852>:	mov    0x8(%rdi,%rcx,8),%r9
<+857>:	mov    0x10(%rdi,%rcx,8),%r10
<+862>:	movd   0x4(%r11),%xmm6
<+868>:	mov    0x18(%rdi,%rcx,8),%r11
<+873>:	movd   0x4(%r9),%xmm14
<+879>:	punpckldq %xmm14,%xmm6
<+884>:	movd   0x4(%r10),%xmm14
<+890>:	movd   0x4(%r11),%xmm15
<+896>:	punpckldq %xmm15,%xmm14
<+901>:	movlhps %xmm14,%xmm6
<+905>:	rcpps  %xmm6,%xmm7
<+908>:	mulps  %xmm7,%xmm6
<+911>:	mulps  %xmm7,%xmm6
<+914>:	addps  %xmm7,%xmm7
<+917>:	mov    (%r8,%rcx,8),%r9
<+921>:	subps  %xmm6,%xmm7
<+924>:	mulps  %xmm2,%xmm7
<+927>:	movdqu 0x10(%r8,%rcx,8),%xmm9
<+934>:	movaps %xmm7,%xmm10
<+938>:	movq   %xmm9,%r11
<+943>:	punpckhqdq %xmm9,%xmm9
<+948>:	movd   %xmm7,0x4(%r9)
<+954>:	movq   %xmm9,%r9
<+959>:	movhlps %xmm7,%xmm10
<+963>:	pshuflw $0xee,%xmm7,%xmm8
<+969>:	mov    0x8(%r8,%rcx,8),%r10
<+974>:	pshuflw $0xee,%xmm10,%xmm11
<+980>:	movd   %xmm8,0x4(%r10)
<+986>:	movd   %xmm10,0x4(%r11)
<+992>:	movd   %xmm11,0x4(%r9)
<+1161>:	movaps %xmm4,%xmm1
<+1182>:	divss  0x4(%rsi),%xmm1
<+1200>:	movss  %xmm1,0x4(%r9)

results[i]->b = arr[i]->b / c;
<+119>:	movss  0x8(%r10),%xmm2
<+135>:	divss  %xmm5,%xmm2
<+150>:	movss  %xmm2,0x8(%r11)
<+163>:	lea    (%r8,%rcx,8),%r9
<+212>:	rcpps  %xmm6,%xmm0
<+215>:	mulps  %xmm0,%xmm6
<+218>:	mulps  %xmm0,%xmm6
<+221>:	addps  %xmm0,%xmm0
<+228>:	subps  %xmm6,%xmm0
<+512>:	mov    (%rdi,%rcx,8),%r10
<+516>:	mov    0x8(%rdi,%rcx,8),%r11
<+521>:	mov    0x10(%rdi,%rcx,8),%r9
<+526>:	movd   0x8(%r10),%xmm6
<+532>:	mov    0x18(%rdi,%rcx,8),%r10
<+537>:	movd   0x8(%r11),%xmm12
<+543>:	punpckldq %xmm12,%xmm6
<+548>:	movd   0x8(%r9),%xmm12
<+554>:	xchg   %ax,%ax
<+556>:	movd   0x8(%r10),%xmm13
<+562>:	punpckldq %xmm13,%xmm12
<+567>:	movlhps %xmm12,%xmm6
<+571>:	mulps  %xmm0,%xmm6
<+574>:	movdqu 0x10(%r8,%rcx,8),%xmm7
<+581>:	mov    (%r8,%rcx,8),%r11
<+585>:	movaps %xmm6,%xmm8
<+589>:	movq   %xmm7,%r10
<+594>:	punpckhqdq %xmm7,%xmm7
<+598>:	movd   %xmm6,0x8(%r11)
<+604>:	movq   %xmm7,%r11
<+609>:	movhlps %xmm6,%xmm8
<+613>:	pshuflw $0xee,%xmm6,%xmm15
<+619>:	mov    0x8(%r8,%rcx,8),%r9
<+628>:	pshuflw $0xee,%xmm8,%xmm6
<+634>:	movd   %xmm15,0x8(%r9)
<+640>:	movd   %xmm8,0x8(%r10)
<+646>:	movd   %xmm6,0x8(%r11)
<+698>:	rcpps  %xmm6,%xmm0
<+701>:	mulps  %xmm0,%xmm6
<+704>:	mulps  %xmm0,%xmm6
<+707>:	addps  %xmm0,%xmm0
<+714>:	subps  %xmm6,%xmm0
<+998>:	mov    (%rdi,%rcx,8),%r10
<+1002>:	mov    0x8(%rdi,%rcx,8),%r11
<+1007>:	mov    0x10(%rdi,%rcx,8),%r9
<+1012>:	movd   0x8(%r10),%xmm6
<+1018>:	mov    0x18(%rdi,%rcx,8),%r10
<+1023>:	movd   0x8(%r11),%xmm12
<+1029>:	punpckldq %xmm12,%xmm6
<+1034>:	movd   0x8(%r9),%xmm12
<+1040>:	movd   0x8(%r10),%xmm13
<+1046>:	punpckldq %xmm13,%xmm12
<+1051>:	movlhps %xmm12,%xmm6
<+1055>:	mulps  %xmm0,%xmm6
<+1058>:	mov    (%r8,%rcx,8),%r11
<+1062>:	movaps %xmm6,%xmm8
<+1066>:	movdqu 0x10(%r8,%rcx,8),%xmm7
<+1073>:	movq   %xmm7,%r10
<+1078>:	punpckhqdq %xmm7,%xmm7
<+1082>:	movd   %xmm6,0x8(%r11)
<+1088>:	movq   %xmm7,%r11
<+1093>:	movhlps %xmm6,%xmm8
<+1097>:	pshuflw $0xee,%xmm6,%xmm15
<+1103>:	mov    0x8(%r8,%rcx,8),%r9
<+1112>:	pshuflw $0xee,%xmm8,%xmm6
<+1118>:	movd   %xmm15,0x8(%r9)
<+1124>:	movd   %xmm8,0x8(%r10)
<+1130>:	movd   %xmm6,0x8(%r11)
<+1177>:	movss  0x8(%rsi),%xmm2
<+1191>:	divss  %xmm5,%xmm2
<+1206>:	movss  %xmm2,0x8(%r9)


33	}
<+1217>:	ret    


for function floating_division_soa:
b, float c, struct pixelArray* arr, struct pixelArray* results, size_t arr_len) {
<+0>:	push   %r12
<+2>:	push   %r13
<+4>:	push   %r14
<+6>:	push   %r15
<+8>:	push   %rbx
<+9>:	push   %rbp
<+10>:	movaps %xmm2,%xmm7
<+13>:	movaps %xmm1,%xmm6
<+16>:	movaps %xmm0,%xmm5
<+208>:	movaps %xmm5,%xmm4
<+211>:	movaps %xmm7,%xmm3
<+214>:	shufps $0x0,%xmm4,%xmm4
<+218>:	movaps %xmm6,%xmm2
<+221>:	rcpps  %xmm4,%xmm1
<+224>:	movaps %xmm1,%xmm0
<+227>:	mulps  %xmm4,%xmm0
<+230>:	mulps  %xmm1,%xmm0
<+233>:	addps  %xmm1,%xmm1
<+236>:	shufps $0x0,%xmm3,%xmm3
<+240>:	subps  %xmm0,%xmm1
<+243>:	rcpps  %xmm3,%xmm0
<+246>:	movaps %xmm0,%xmm8
<+250>:	mulps  %xmm3,%xmm8
<+254>:	mulps  %xmm0,%xmm8
<+258>:	addps  %xmm0,%xmm0
<+261>:	shufps $0x0,%xmm2,%xmm2
<+265>:	subps  %xmm8,%xmm0
<+421>:	movaps %xmm5,%xmm4
<+424>:	movaps %xmm7,%xmm3
<+427>:	shufps $0x0,%xmm4,%xmm4
<+431>:	movaps %xmm6,%xmm2
<+434>:	rcpps  %xmm4,%xmm1
<+437>:	movaps %xmm1,%xmm0
<+440>:	mulps  %xmm4,%xmm0
<+443>:	mulps  %xmm1,%xmm0
<+446>:	addps  %xmm1,%xmm1
<+449>:	shufps $0x0,%xmm3,%xmm3
<+453>:	subps  %xmm0,%xmm1
<+456>:	rcpps  %xmm3,%xmm0
<+459>:	movaps %xmm0,%xmm8
<+463>:	mulps  %xmm3,%xmm8
<+467>:	mulps  %xmm0,%xmm8
<+471>:	addps  %xmm0,%xmm0
<+474>:	shufps $0x0,%xmm2,%xmm2
<+478>:	subps  %xmm8,%xmm0


 

i = 0; i < arr_len; ++i) {
<+19>:	test   %rdx,%rdx
<+22>:	jbe    0x87e <floating_division_soa+942>
<+50>:	cmp    $0x8,%rdx
<+54>:	jb     0x894 <floating_division_soa+964>
<+60>:	mov    %r11,%rax
<+63>:	and    $0xf,%rax
<+67>:	je     0x52c <floating_division_soa+92>
<+69>:	test   $0x3,%rax
<+75>:	jne    0x88d <floating_division_soa+957>
<+81>:	neg    %rax
<+84>:	add    $0x10,%rax
<+88>:	shr    $0x2,%rax
<+92>:	lea    0x8(%rax),%rcx
<+96>:	cmp    %rcx,%rdx
<+99>:	jb     0x88d <floating_division_soa+957>
<+105>:	mov    %rdx,%r9
<+108>:	xor    %ebx,%ebx
<+110>:	sub    %rax,%r9
<+113>:	xor    %ecx,%ecx
<+115>:	and    $0x7,%r9
<+119>:	neg    %r9
<+122>:	add    %rdx,%r9
<+125>:	test   %rax,%rax
<+128>:	jbe    0x58c <floating_division_soa+188>
<+149>:	inc    %ebx
<+180>:	inc    %rcx
<+183>:	cmp    %rax,%rbx
<+186>:	jb     0x552 <floating_division_soa+130>
<+195>:	test   $0xf,%rbx
<+202>:	je     0x675 <floating_division_soa+421>
<+278>:	add    $0x8,%ecx
<+403>:	add    $0x8,%rax
<+407>:	cmp    %r9,%rcx
<+410>:	jb     0x5dd <floating_division_soa+269>
<+416>:	jmp    0x745 <floating_division_soa+629>
<+491>:	add    $0x8,%ecx
<+616>:	add    $0x8,%rax
<+620>:	cmp    %r9,%rcx
<+623>:	jb     0x6b2 <floating_division_soa+482>
<+629>:	lea    0x1(%r9),%rax
<+633>:	cmp    %rdx,%rax
<+636>:	ja     0x87e <floating_division_soa+942>
<+642>:	sub    %r9,%rdx
<+645>:	cmp    $0x4,%rdx
<+649>:	jb     0x889 <floating_division_soa+953>
<+658>:	mov    %rdx,%rax
<+661>:	movl   $0x0,-0x8(%rsp)
<+669>:	and    $0xfffffffffffffffc,%rax
<+673>:	mov    %r15,-0x18(%rsp)
<+678>:	mov    %rdx,-0x10(%rsp)
<+687>:	mov    %rcx,%rdx
<+694>:	mov    -0x8(%rsp),%r15d
<+716>:	xor    %ebx,%ebx
<+718>:	xchg   %ax,%ax
<+743>:	add    $0x4,%r15d
<+829>:	add    $0x4,%rbx
<+833>:	cmp    %rax,%r15
<+836>:	jb     0x7a0 <floating_division_soa+720>
<+838>:	mov    -0x18(%rsp),%r15
<+843>:	mov    -0x10(%rsp),%rdx
<+848>:	movslq %eax,%rcx
<+851>:	mov    %eax,%ebx
<+853>:	mov    %eax,%eax
<+855>:	cmp    %rdx,%rax
<+858>:	jae    0x87e <floating_division_soa+942>
<+906>:	inc    %ebx
<+934>:	inc    %rcx
<+937>:	cmp    %rdx,%rbx
<+940>:	jb     0x848 <floating_division_soa+888>
<+953>:	xor    %eax,%eax
<+955>:	jmp    0x820 <floating_division_soa+848>
<+957>:	xor    %r9d,%r9d
<+960>:	xor    %eax,%eax
<+962>:	jmp    0x820 <floating_division_soa+848>
<+964>:	cmp    $0x1,%rdx
<+968>:	jae    0x88d <floating_division_soa+957>
<+970>:	jmp    0x87e <floating_division_soa+942>
<+972>:	nopl   0x0(%rax)

results->r[i] = arr->r[i] / a;
<+28>:	mov    (%rsi),%r15
<+39>:	mov    (%rdi),%r14
<+130>:	movss  (%r14,%rcx,4),%xmm0
<+139>:	divss  %xmm5,%xmm0
<+143>:	movss  %xmm0,(%r15,%rcx,4)
<+188>:	mov    %eax,%ecx
<+269>:	movups (%r14,%rax,4),%xmm8
<+274>:	mulps  %xmm1,%xmm8
<+281>:	movups %xmm8,(%r15,%rax,4)
<+335>:	movups 0x10(%r14,%rax,4),%xmm12
<+341>:	mulps  %xmm1,%xmm12
<+345>:	movups %xmm12,0x10(%r15,%rax,4)
<+482>:	movups (%r14,%rax,4),%xmm8
<+487>:	mulps  %xmm1,%xmm8
<+494>:	movups %xmm8,(%r15,%rax,4)
<+548>:	movups 0x10(%r14,%rax,4),%xmm12
<+554>:	mulps  %xmm1,%xmm12
<+558>:	movups %xmm12,0x10(%r15,%rax,4)
<+655>:	movslq %r9d,%rbx
<+683>:	lea    (%r15,%rbx,4),%rcx
<+690>:	lea    (%r14,%rbx,4),%rbp
<+720>:	rcpps  %xmm4,%xmm1
<+723>:	movups 0x0(%rbp,%rbx,4),%xmm8
<+733>:	movaps %xmm4,%xmm0
<+740>:	mulps  %xmm1,%xmm0
<+751>:	mulps  %xmm1,%xmm0
<+754>:	addps  %xmm1,%xmm1
<+765>:	subps  %xmm0,%xmm1
<+772>:	mulps  %xmm1,%xmm8
<+776>:	movups %xmm8,(%rdx,%rbx,4)
<+860>:	movslq %r9d,%rbp
<+863>:	lea    (%r15,%rbp,4),%rax
<+867>:	lea    (%r14,%rbp,4),%r9
<+888>:	movss  (%r9,%rcx,4),%xmm0
<+897>:	divss  %xmm5,%xmm0
<+901>:	movss  %xmm0,(%rax,%rcx,4)

results->g[i] = b / arr->g[i];
<+31>:	mov    0x8(%rsi),%r13
<+42>:	mov    0x8(%rdi),%r12
<+136>:	movaps %xmm6,%xmm1
<+151>:	divss  (%r12,%rcx,4),%xmm1
<+157>:	movss  %xmm1,0x0(%r13,%rcx,4)
<+190>:	lea    0x0(%r13,%rax,4),%rbx
<+286>:	movups (%r12,%rax,4),%xmm9
<+291>:	rcpps  %xmm9,%xmm10
<+295>:	mulps  %xmm10,%xmm9
<+299>:	mulps  %xmm10,%xmm9
<+303>:	addps  %xmm10,%xmm10
<+307>:	subps  %xmm9,%xmm10
<+311>:	mulps  %xmm2,%xmm10
<+315>:	movups %xmm10,0x0(%r13,%rax,4)
<+351>:	movups 0x10(%r12,%rax,4),%xmm13
<+357>:	rcpps  %xmm13,%xmm14
<+361>:	mulps  %xmm14,%xmm13
<+365>:	mulps  %xmm14,%xmm13
<+369>:	addps  %xmm14,%xmm14
<+373>:	subps  %xmm13,%xmm14
<+377>:	mulps  %xmm2,%xmm14
<+381>:	movups %xmm14,0x10(%r13,%rax,4)
<+499>:	movups (%r12,%rax,4),%xmm9
<+504>:	rcpps  %xmm9,%xmm10
<+508>:	mulps  %xmm10,%xmm9
<+512>:	mulps  %xmm10,%xmm9
<+516>:	addps  %xmm10,%xmm10
<+520>:	subps  %xmm9,%xmm10
<+524>:	mulps  %xmm2,%xmm10
<+528>:	movups %xmm10,0x0(%r13,%rax,4)
<+564>:	movups 0x10(%r12,%rax,4),%xmm13
<+570>:	rcpps  %xmm13,%xmm14
<+574>:	mulps  %xmm14,%xmm13
<+578>:	mulps  %xmm14,%xmm13
<+582>:	addps  %xmm14,%xmm14
<+586>:	subps  %xmm13,%xmm14
<+590>:	mulps  %xmm2,%xmm14
<+594>:	movups %xmm14,0x10(%r13,%rax,4)
<+699>:	lea    0x0(%r13,%rbx,4),%rsi
<+704>:	lea    (%r12,%rbx,4),%rdi
<+781>:	movups (%rdi,%rbx,4),%xmm9
<+786>:	rcpps  %xmm9,%xmm10
<+790>:	mulps  %xmm10,%xmm9
<+794>:	mulps  %xmm10,%xmm9
<+798>:	addps  %xmm10,%xmm10
<+802>:	subps  %xmm9,%xmm10
<+806>:	mulps  %xmm2,%xmm10
<+810>:	movups %xmm10,(%rsi,%rbx,4)
<+871>:	lea    0x0(%r13,%rbp,4),%r8
<+876>:	lea    (%r12,%rbp,4),%rdi
<+894>:	movaps %xmm6,%xmm1
<+908>:	divss  (%rdi,%rcx,4),%xmm1
<+913>:	movss  %xmm1,(%r8,%rcx,4)

results->b[i] = arr->b[i] / c;
<+35>:	mov    0x10(%rsi),%r11
<+46>:	mov    0x10(%rdi),%r10
<+164>:	movss  (%r10,%rcx,4),%xmm2
<+170>:	divss  %xmm7,%xmm2
<+174>:	movss  %xmm2,(%r11,%rcx,4)
<+321>:	movups (%r10,%rax,4),%xmm11
<+326>:	mulps  %xmm0,%xmm11
<+330>:	movups %xmm11,(%r11,%rax,4)
<+387>:	movups 0x10(%r10,%rax,4),%xmm15
<+393>:	mulps  %xmm0,%xmm15
<+397>:	movups %xmm15,0x10(%r11,%rax,4)
<+534>:	movups (%r10,%rax,4),%xmm11
<+539>:	mulps  %xmm0,%xmm11
<+543>:	movups %xmm11,(%r11,%rax,4)
<+600>:	movups 0x10(%r10,%rax,4),%xmm15
<+606>:	mulps  %xmm0,%xmm15
<+610>:	movups %xmm15,0x10(%r11,%rax,4)
<+708>:	lea    (%r11,%rbx,4),%rcx
<+712>:	lea    (%r10,%rbx,4),%r8
<+729>:	rcpps  %xmm3,%xmm12
<+736>:	movaps %xmm3,%xmm11
<+747>:	mulps  %xmm12,%xmm11
<+757>:	mulps  %xmm12,%xmm11
<+761>:	addps  %xmm12,%xmm12
<+768>:	subps  %xmm11,%xmm12
<+815>:	movups (%r8,%rbx,4),%xmm13
<+820>:	mulps  %xmm12,%xmm13
<+824>:	movups %xmm13,(%rcx,%rbx,4)
<+880>:	lea    (%r11,%rbp,4),%rsi
<+884>:	lea    (%r10,%rbp,4),%rbp
<+919>:	movss  0x0(%rbp,%rcx,4),%xmm2
<+925>:	divss  %xmm7,%xmm2
<+929>:	movss  %xmm2,(%rsi,%rcx,4)


44	}
<+942>:	pop    %rbp
<+943>:	pop    %rbx
<+944>:	pop    %r15
<+946>:	pop    %r14
<+948>:	pop    %r13
<+950>:	pop    %r12
<+952>:	ret    


