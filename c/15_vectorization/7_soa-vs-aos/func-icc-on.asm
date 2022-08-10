for function floating_division_aos:
b, float c, struct pixel** arr, struct pixel** results, size_t arr_len) {
<+0>:	mov    r8,rsi
<+3>:	movaps xmm5,xmm2
<+6>:	movaps xmm4,xmm1
<+9>:	movaps xmm3,xmm0
<+180>:	movaps xmm0,xmm3
<+183>:	movaps xmm6,xmm5
<+186>:	shufps xmm0,xmm0,0x0
<+190>:	movaps xmm2,xmm4
<+205>:	shufps xmm6,xmm6,0x0
<+224>:	shufps xmm2,xmm2,0x0
<+666>:	movaps xmm0,xmm3
<+669>:	movaps xmm6,xmm5
<+672>:	shufps xmm0,xmm0,0x0
<+676>:	movaps xmm2,xmm4
<+691>:	shufps xmm6,xmm6,0x0
<+710>:	shufps xmm2,xmm2,0x0




are specific for the compiler and platform in use. So the best bet is to look at compiler's documentation.


i = 0; i < arr_len; ++i) {
<+12>:	test   rdx,rdx
<+15>:	jbe    0x4c1 <floating_division_aos+1217>
<+21>:	cmp    rdx,0x4
<+25>:	jb     0x4c2 <floating_division_aos+1218>
<+31>:	mov    rcx,rdi
<+34>:	and    rcx,0xf
<+38>:	je     0x3a <floating_division_aos+58>
<+40>:	test   rcx,0x7
<+47>:	jne    0x4c2 <floating_division_aos+1218>
<+53>:	mov    ecx,0x1
<+58>:	lea    rax,[rcx+0x4]
<+62>:	cmp    rdx,rax
<+65>:	jb     0x4c2 <floating_division_aos+1218>
<+71>:	mov    rsi,rdx
<+74>:	xor    r9d,r9d
<+77>:	sub    rsi,rcx
<+80>:	xor    eax,eax
<+82>:	and    rsi,0x3
<+86>:	neg    rsi
<+89>:	add    rsi,rdx
<+92>:	test   rcx,rcx
<+95>:	jbe    0xa1 <floating_division_aos+161>
<+108>:	inc    r9d
<+111>:	inc    rax
<+156>:	cmp    r9,rcx
<+159>:	jb     0x61 <floating_division_aos+97>
<+167>:	test   r9,0xf
<+174>:	je     0x29a <floating_division_aos+666>
<+235>:	add    eax,0x4
<+624>:	add    rcx,0x4
<+652>:	cmp    rax,rsi
<+655>:	jb     0xe7 <floating_division_aos+231>
<+661>:	jmp    0x479 <floating_division_aos+1145>
<+721>:	add    eax,0x4
<+1108>:	add    rcx,0x4
<+1136>:	cmp    rax,rsi
<+1139>:	jb     0x2cd <floating_division_aos+717>
<+1145>:	movsxd rax,esi
<+1148>:	mov    ecx,esi
<+1150>:	mov    esi,esi
<+1152>:	cmp    rsi,rdx
<+1155>:	jae    0x4c1 <floating_division_aos+1217>
<+1168>:	inc    ecx
<+1170>:	inc    rax
<+1212>:	cmp    rcx,rdx
<+1215>:	jb     0x485 <floating_division_aos+1157>
<+1218>:	xor    esi,esi
<+1220>:	jmp    0x479 <floating_division_aos+1145>
<+1222>:	nop    DWORD PTR [rax]
<+1225>:	nop    DWORD PTR [rax+0x0]

results[i]->r = arr[i]->r / a;
<+97>:	mov    r10,QWORD PTR [rdi+rax*8]
<+104>:	mov    r11,QWORD PTR [r8+rax*8]
<+114>:	movss  xmm0,DWORD PTR [r10]
<+131>:	divss  xmm0,xmm3
<+139>:	movss  DWORD PTR [r11],xmm0
<+161>:	mov    eax,ecx
<+193>:	rcpps  xmm1,xmm0
<+196>:	mulps  xmm0,xmm1
<+199>:	mulps  xmm0,xmm1
<+202>:	addps  xmm1,xmm1
<+209>:	subps  xmm1,xmm0
<+231>:	mov    r9,QWORD PTR [rdi+rcx*8]
<+238>:	mov    r10,QWORD PTR [rdi+rcx*8+0x8]
<+243>:	mov    r11,QWORD PTR [rdi+rcx*8+0x10]
<+248>:	movd   xmm9,DWORD PTR [r9]
<+253>:	mov    r9,QWORD PTR [rdi+rcx*8+0x18]
<+258>:	movd   xmm6,DWORD PTR [r10]
<+263>:	movd   xmm8,DWORD PTR [r11]
<+268>:	movd   xmm7,DWORD PTR [r9]
<+273>:	punpckldq xmm9,xmm6
<+278>:	punpckldq xmm8,xmm7
<+283>:	movlhps xmm9,xmm8
<+287>:	mulps  xmm9,xmm1
<+291>:	movdqu xmm11,XMMWORD PTR [r8+rcx*8+0x10]
<+298>:	mov    r10,QWORD PTR [r8+rcx*8]
<+302>:	movaps xmm12,xmm9
<+306>:	movq   r9,xmm11
<+311>:	punpckhqdq xmm11,xmm11
<+316>:	movd   DWORD PTR [r10],xmm9
<+321>:	movq   r10,xmm11
<+326>:	movhlps xmm12,xmm9
<+330>:	pshuflw xmm10,xmm9,0xee
<+336>:	mov    r11,QWORD PTR [r8+rcx*8+0x8]
<+341>:	pshuflw xmm13,xmm12,0xee
<+347>:	movd   DWORD PTR [r11],xmm10
<+352>:	movd   DWORD PTR [r9],xmm12
<+357>:	movd   DWORD PTR [r10],xmm13
<+679>:	rcpps  xmm1,xmm0
<+682>:	mulps  xmm0,xmm1
<+685>:	mulps  xmm0,xmm1
<+688>:	addps  xmm1,xmm1
<+695>:	subps  xmm1,xmm0
<+717>:	mov    r9,QWORD PTR [rdi+rcx*8]
<+724>:	mov    r10,QWORD PTR [rdi+rcx*8+0x8]
<+729>:	mov    r11,QWORD PTR [rdi+rcx*8+0x10]
<+734>:	movd   xmm9,DWORD PTR [r9]
<+739>:	mov    r9,QWORD PTR [rdi+rcx*8+0x18]
<+744>:	movd   xmm6,DWORD PTR [r10]
<+749>:	movd   xmm8,DWORD PTR [r11]
<+754>:	movd   xmm7,DWORD PTR [r9]
<+759>:	punpckldq xmm9,xmm6
<+764>:	punpckldq xmm8,xmm7
<+769>:	movlhps xmm9,xmm8
<+773>:	mulps  xmm9,xmm1
<+777>:	mov    r10,QWORD PTR [r8+rcx*8]
<+781>:	movaps xmm12,xmm9
<+785>:	movdqu xmm11,XMMWORD PTR [r8+rcx*8+0x10]
<+792>:	movq   r9,xmm11
<+797>:	punpckhqdq xmm11,xmm11
<+802>:	movd   DWORD PTR [r10],xmm9
<+807>:	movq   r10,xmm11
<+812>:	movhlps xmm12,xmm9
<+816>:	pshuflw xmm10,xmm9,0xee
<+822>:	mov    r11,QWORD PTR [r8+rcx*8+0x8]
<+827>:	pshuflw xmm13,xmm12,0xee
<+833>:	movd   DWORD PTR [r11],xmm10
<+838>:	movd   DWORD PTR [r9],xmm12
<+843>:	movd   DWORD PTR [r10],xmm13
<+1157>:	mov    rsi,QWORD PTR [rdi+rax*8]
<+1164>:	mov    r9,QWORD PTR [r8+rax*8]
<+1173>:	movss  xmm0,DWORD PTR [rsi]
<+1187>:	divss  xmm0,xmm3
<+1195>:	movss  DWORD PTR [r9],xmm0

results[i]->g = b / arr[i]->g;
<+101>:	movaps xmm1,xmm4
<+125>:	divss  xmm1,DWORD PTR [r10+0x4]
<+144>:	movss  DWORD PTR [r11+0x4],xmm1
<+362>:	mov    r11,QWORD PTR [rdi+rcx*8]
<+366>:	mov    r9,QWORD PTR [rdi+rcx*8+0x8]
<+371>:	mov    r10,QWORD PTR [rdi+rcx*8+0x10]
<+376>:	movd   xmm6,DWORD PTR [r11+0x4]
<+382>:	mov    r11,QWORD PTR [rdi+rcx*8+0x18]
<+387>:	movd   xmm14,DWORD PTR [r9+0x4]
<+393>:	punpckldq xmm6,xmm14
<+398>:	movd   xmm14,DWORD PTR [r10+0x4]
<+404>:	movd   xmm15,DWORD PTR [r11+0x4]
<+410>:	punpckldq xmm14,xmm15
<+415>:	movlhps xmm6,xmm14
<+419>:	rcpps  xmm7,xmm6
<+422>:	movdqu xmm9,XMMWORD PTR [r8+rcx*8+0x10]
<+429>:	mulps  xmm6,xmm7
<+432>:	movq   r11,xmm9
<+437>:	punpckhqdq xmm9,xmm9
<+442>:	mulps  xmm6,xmm7
<+445>:	addps  xmm7,xmm7
<+448>:	mov    r9,QWORD PTR [r8+rcx*8]
<+452>:	subps  xmm7,xmm6
<+455>:	mulps  xmm7,xmm2
<+458>:	movd   DWORD PTR [r9+0x4],xmm7
<+464>:	movaps xmm10,xmm7
<+468>:	movq   r9,xmm9
<+473>:	movhlps xmm10,xmm7
<+477>:	pshuflw xmm8,xmm7,0xee
<+483>:	mov    r10,QWORD PTR [r8+rcx*8+0x8]
<+488>:	pshuflw xmm11,xmm10,0xee
<+494>:	movd   DWORD PTR [r10+0x4],xmm8
<+500>:	movd   DWORD PTR [r11+0x4],xmm10
<+506>:	movd   DWORD PTR [r9+0x4],xmm11
<+848>:	mov    r11,QWORD PTR [rdi+rcx*8]
<+852>:	mov    r9,QWORD PTR [rdi+rcx*8+0x8]
<+857>:	mov    r10,QWORD PTR [rdi+rcx*8+0x10]
<+862>:	movd   xmm6,DWORD PTR [r11+0x4]
<+868>:	mov    r11,QWORD PTR [rdi+rcx*8+0x18]
<+873>:	movd   xmm14,DWORD PTR [r9+0x4]
<+879>:	punpckldq xmm6,xmm14
<+884>:	movd   xmm14,DWORD PTR [r10+0x4]
<+890>:	movd   xmm15,DWORD PTR [r11+0x4]
<+896>:	punpckldq xmm14,xmm15
<+901>:	movlhps xmm6,xmm14
<+905>:	rcpps  xmm7,xmm6
<+908>:	mulps  xmm6,xmm7
<+911>:	mulps  xmm6,xmm7
<+914>:	addps  xmm7,xmm7
<+917>:	mov    r9,QWORD PTR [r8+rcx*8]
<+921>:	subps  xmm7,xmm6
<+924>:	mulps  xmm7,xmm2
<+927>:	movdqu xmm9,XMMWORD PTR [r8+rcx*8+0x10]
<+934>:	movaps xmm10,xmm7
<+938>:	movq   r11,xmm9
<+943>:	punpckhqdq xmm9,xmm9
<+948>:	movd   DWORD PTR [r9+0x4],xmm7
<+954>:	movq   r9,xmm9
<+959>:	movhlps xmm10,xmm7
<+963>:	pshuflw xmm8,xmm7,0xee
<+969>:	mov    r10,QWORD PTR [r8+rcx*8+0x8]
<+974>:	pshuflw xmm11,xmm10,0xee
<+980>:	movd   DWORD PTR [r10+0x4],xmm8
<+986>:	movd   DWORD PTR [r11+0x4],xmm10
<+992>:	movd   DWORD PTR [r9+0x4],xmm11
<+1161>:	movaps xmm1,xmm4
<+1182>:	divss  xmm1,DWORD PTR [rsi+0x4]
<+1200>:	movss  DWORD PTR [r9+0x4],xmm1

results[i]->b = arr[i]->b / c;
<+119>:	movss  xmm2,DWORD PTR [r10+0x8]
<+135>:	divss  xmm2,xmm5
<+150>:	movss  DWORD PTR [r11+0x8],xmm2
<+163>:	lea    r9,[r8+rcx*8]
<+212>:	rcpps  xmm0,xmm6
<+215>:	mulps  xmm6,xmm0
<+218>:	mulps  xmm6,xmm0
<+221>:	addps  xmm0,xmm0
<+228>:	subps  xmm0,xmm6
<+512>:	mov    r10,QWORD PTR [rdi+rcx*8]
<+516>:	mov    r11,QWORD PTR [rdi+rcx*8+0x8]
<+521>:	mov    r9,QWORD PTR [rdi+rcx*8+0x10]
<+526>:	movd   xmm6,DWORD PTR [r10+0x8]
<+532>:	mov    r10,QWORD PTR [rdi+rcx*8+0x18]
<+537>:	movd   xmm12,DWORD PTR [r11+0x8]
<+543>:	punpckldq xmm6,xmm12
<+548>:	movd   xmm12,DWORD PTR [r9+0x8]
<+554>:	xchg   ax,ax
<+556>:	movd   xmm13,DWORD PTR [r10+0x8]
<+562>:	punpckldq xmm12,xmm13
<+567>:	movlhps xmm6,xmm12
<+571>:	mulps  xmm6,xmm0
<+574>:	movdqu xmm7,XMMWORD PTR [r8+rcx*8+0x10]
<+581>:	mov    r11,QWORD PTR [r8+rcx*8]
<+585>:	movaps xmm8,xmm6
<+589>:	movq   r10,xmm7
<+594>:	punpckhqdq xmm7,xmm7
<+598>:	movd   DWORD PTR [r11+0x8],xmm6
<+604>:	movq   r11,xmm7
<+609>:	movhlps xmm8,xmm6
<+613>:	pshuflw xmm15,xmm6,0xee
<+619>:	mov    r9,QWORD PTR [r8+rcx*8+0x8]
<+628>:	pshuflw xmm6,xmm8,0xee
<+634>:	movd   DWORD PTR [r9+0x8],xmm15
<+640>:	movd   DWORD PTR [r10+0x8],xmm8
<+646>:	movd   DWORD PTR [r11+0x8],xmm6
<+698>:	rcpps  xmm0,xmm6
<+701>:	mulps  xmm6,xmm0
<+704>:	mulps  xmm6,xmm0
<+707>:	addps  xmm0,xmm0
<+714>:	subps  xmm0,xmm6
<+998>:	mov    r10,QWORD PTR [rdi+rcx*8]
<+1002>:	mov    r11,QWORD PTR [rdi+rcx*8+0x8]
<+1007>:	mov    r9,QWORD PTR [rdi+rcx*8+0x10]
<+1012>:	movd   xmm6,DWORD PTR [r10+0x8]
<+1018>:	mov    r10,QWORD PTR [rdi+rcx*8+0x18]
<+1023>:	movd   xmm12,DWORD PTR [r11+0x8]
<+1029>:	punpckldq xmm6,xmm12
<+1034>:	movd   xmm12,DWORD PTR [r9+0x8]
<+1040>:	movd   xmm13,DWORD PTR [r10+0x8]
<+1046>:	punpckldq xmm12,xmm13
<+1051>:	movlhps xmm6,xmm12
<+1055>:	mulps  xmm6,xmm0
<+1058>:	mov    r11,QWORD PTR [r8+rcx*8]
<+1062>:	movaps xmm8,xmm6
<+1066>:	movdqu xmm7,XMMWORD PTR [r8+rcx*8+0x10]
<+1073>:	movq   r10,xmm7
<+1078>:	punpckhqdq xmm7,xmm7
<+1082>:	movd   DWORD PTR [r11+0x8],xmm6
<+1088>:	movq   r11,xmm7
<+1093>:	movhlps xmm8,xmm6
<+1097>:	pshuflw xmm15,xmm6,0xee
<+1103>:	mov    r9,QWORD PTR [r8+rcx*8+0x8]
<+1112>:	pshuflw xmm6,xmm8,0xee
<+1118>:	movd   DWORD PTR [r9+0x8],xmm15
<+1124>:	movd   DWORD PTR [r10+0x8],xmm8
<+1130>:	movd   DWORD PTR [r11+0x8],xmm6
<+1177>:	movss  xmm2,DWORD PTR [rsi+0x8]
<+1191>:	divss  xmm2,xmm5
<+1206>:	movss  DWORD PTR [r9+0x8],xmm2


33	}
<+1217>:	ret    


for function floating_division_soa:
b, float c, struct pixelArray* arr, struct pixelArray* results, size_t arr_len) {
<+0>:	push   r12
<+2>:	push   r13
<+4>:	push   r14
<+6>:	push   r15
<+8>:	push   rbx
<+9>:	push   rbp
<+10>:	movaps xmm7,xmm2
<+13>:	movaps xmm6,xmm1
<+16>:	movaps xmm5,xmm0
<+208>:	movaps xmm4,xmm5
<+211>:	movaps xmm3,xmm7
<+214>:	shufps xmm4,xmm4,0x0
<+218>:	movaps xmm2,xmm6
<+221>:	rcpps  xmm1,xmm4
<+224>:	movaps xmm0,xmm1
<+227>:	mulps  xmm0,xmm4
<+230>:	mulps  xmm0,xmm1
<+233>:	addps  xmm1,xmm1
<+236>:	shufps xmm3,xmm3,0x0
<+240>:	subps  xmm1,xmm0
<+243>:	rcpps  xmm0,xmm3
<+246>:	movaps xmm8,xmm0
<+250>:	mulps  xmm8,xmm3
<+254>:	mulps  xmm8,xmm0
<+258>:	addps  xmm0,xmm0
<+261>:	shufps xmm2,xmm2,0x0
<+265>:	subps  xmm0,xmm8
<+421>:	movaps xmm4,xmm5
<+424>:	movaps xmm3,xmm7
<+427>:	shufps xmm4,xmm4,0x0
<+431>:	movaps xmm2,xmm6
<+434>:	rcpps  xmm1,xmm4
<+437>:	movaps xmm0,xmm1
<+440>:	mulps  xmm0,xmm4
<+443>:	mulps  xmm0,xmm1
<+446>:	addps  xmm1,xmm1
<+449>:	shufps xmm3,xmm3,0x0
<+453>:	subps  xmm1,xmm0
<+456>:	rcpps  xmm0,xmm3
<+459>:	movaps xmm8,xmm0
<+463>:	mulps  xmm8,xmm3
<+467>:	mulps  xmm8,xmm0
<+471>:	addps  xmm0,xmm0
<+474>:	shufps xmm2,xmm2,0x0
<+478>:	subps  xmm0,xmm8


 

i = 0; i < arr_len; ++i) {
<+19>:	test   rdx,rdx
<+22>:	jbe    0x87e <floating_division_soa+942>
<+50>:	cmp    rdx,0x8
<+54>:	jb     0x894 <floating_division_soa+964>
<+60>:	mov    rax,r11
<+63>:	and    rax,0xf
<+67>:	je     0x52c <floating_division_soa+92>
<+69>:	test   rax,0x3
<+75>:	jne    0x88d <floating_division_soa+957>
<+81>:	neg    rax
<+84>:	add    rax,0x10
<+88>:	shr    rax,0x2
<+92>:	lea    rcx,[rax+0x8]
<+96>:	cmp    rdx,rcx
<+99>:	jb     0x88d <floating_division_soa+957>
<+105>:	mov    r9,rdx
<+108>:	xor    ebx,ebx
<+110>:	sub    r9,rax
<+113>:	xor    ecx,ecx
<+115>:	and    r9,0x7
<+119>:	neg    r9
<+122>:	add    r9,rdx
<+125>:	test   rax,rax
<+128>:	jbe    0x58c <floating_division_soa+188>
<+149>:	inc    ebx
<+180>:	inc    rcx
<+183>:	cmp    rbx,rax
<+186>:	jb     0x552 <floating_division_soa+130>
<+195>:	test   rbx,0xf
<+202>:	je     0x675 <floating_division_soa+421>
<+278>:	add    ecx,0x8
<+403>:	add    rax,0x8
<+407>:	cmp    rcx,r9
<+410>:	jb     0x5dd <floating_division_soa+269>
<+416>:	jmp    0x745 <floating_division_soa+629>
<+491>:	add    ecx,0x8
<+616>:	add    rax,0x8
<+620>:	cmp    rcx,r9
<+623>:	jb     0x6b2 <floating_division_soa+482>
<+629>:	lea    rax,[r9+0x1]
<+633>:	cmp    rax,rdx
<+636>:	ja     0x87e <floating_division_soa+942>
<+642>:	sub    rdx,r9
<+645>:	cmp    rdx,0x4
<+649>:	jb     0x889 <floating_division_soa+953>
<+658>:	mov    rax,rdx
<+661>:	mov    DWORD PTR [rsp-0x8],0x0
<+669>:	and    rax,0xfffffffffffffffc
<+673>:	mov    QWORD PTR [rsp-0x18],r15
<+678>:	mov    QWORD PTR [rsp-0x10],rdx
<+687>:	mov    rdx,rcx
<+694>:	mov    r15d,DWORD PTR [rsp-0x8]
<+716>:	xor    ebx,ebx
<+718>:	xchg   ax,ax
<+743>:	add    r15d,0x4
<+829>:	add    rbx,0x4
<+833>:	cmp    r15,rax
<+836>:	jb     0x7a0 <floating_division_soa+720>
<+838>:	mov    r15,QWORD PTR [rsp-0x18]
<+843>:	mov    rdx,QWORD PTR [rsp-0x10]
<+848>:	movsxd rcx,eax
<+851>:	mov    ebx,eax
<+853>:	mov    eax,eax
<+855>:	cmp    rax,rdx
<+858>:	jae    0x87e <floating_division_soa+942>
<+906>:	inc    ebx
<+934>:	inc    rcx
<+937>:	cmp    rbx,rdx
<+940>:	jb     0x848 <floating_division_soa+888>
<+953>:	xor    eax,eax
<+955>:	jmp    0x820 <floating_division_soa+848>
<+957>:	xor    r9d,r9d
<+960>:	xor    eax,eax
<+962>:	jmp    0x820 <floating_division_soa+848>
<+964>:	cmp    rdx,0x1
<+968>:	jae    0x88d <floating_division_soa+957>
<+970>:	jmp    0x87e <floating_division_soa+942>
<+972>:	nop    DWORD PTR [rax+0x0]

results->r[i] = arr->r[i] / a;
<+28>:	mov    r15,QWORD PTR [rsi]
<+39>:	mov    r14,QWORD PTR [rdi]
<+130>:	movss  xmm0,DWORD PTR [r14+rcx*4]
<+139>:	divss  xmm0,xmm5
<+143>:	movss  DWORD PTR [r15+rcx*4],xmm0
<+188>:	mov    ecx,eax
<+269>:	movups xmm8,XMMWORD PTR [r14+rax*4]
<+274>:	mulps  xmm8,xmm1
<+281>:	movups XMMWORD PTR [r15+rax*4],xmm8
<+335>:	movups xmm12,XMMWORD PTR [r14+rax*4+0x10]
<+341>:	mulps  xmm12,xmm1
<+345>:	movups XMMWORD PTR [r15+rax*4+0x10],xmm12
<+482>:	movups xmm8,XMMWORD PTR [r14+rax*4]
<+487>:	mulps  xmm8,xmm1
<+494>:	movups XMMWORD PTR [r15+rax*4],xmm8
<+548>:	movups xmm12,XMMWORD PTR [r14+rax*4+0x10]
<+554>:	mulps  xmm12,xmm1
<+558>:	movups XMMWORD PTR [r15+rax*4+0x10],xmm12
<+655>:	movsxd rbx,r9d
<+683>:	lea    rcx,[r15+rbx*4]
<+690>:	lea    rbp,[r14+rbx*4]
<+720>:	rcpps  xmm1,xmm4
<+723>:	movups xmm8,XMMWORD PTR [rbp+rbx*4+0x0]
<+733>:	movaps xmm0,xmm4
<+740>:	mulps  xmm0,xmm1
<+751>:	mulps  xmm0,xmm1
<+754>:	addps  xmm1,xmm1
<+765>:	subps  xmm1,xmm0
<+772>:	mulps  xmm8,xmm1
<+776>:	movups XMMWORD PTR [rdx+rbx*4],xmm8
<+860>:	movsxd rbp,r9d
<+863>:	lea    rax,[r15+rbp*4]
<+867>:	lea    r9,[r14+rbp*4]
<+888>:	movss  xmm0,DWORD PTR [r9+rcx*4]
<+897>:	divss  xmm0,xmm5
<+901>:	movss  DWORD PTR [rax+rcx*4],xmm0

results->g[i] = b / arr->g[i];
<+31>:	mov    r13,QWORD PTR [rsi+0x8]
<+42>:	mov    r12,QWORD PTR [rdi+0x8]
<+136>:	movaps xmm1,xmm6
<+151>:	divss  xmm1,DWORD PTR [r12+rcx*4]
<+157>:	movss  DWORD PTR [r13+rcx*4+0x0],xmm1
<+190>:	lea    rbx,[r13+rax*4+0x0]
<+286>:	movups xmm9,XMMWORD PTR [r12+rax*4]
<+291>:	rcpps  xmm10,xmm9
<+295>:	mulps  xmm9,xmm10
<+299>:	mulps  xmm9,xmm10
<+303>:	addps  xmm10,xmm10
<+307>:	subps  xmm10,xmm9
<+311>:	mulps  xmm10,xmm2
<+315>:	movups XMMWORD PTR [r13+rax*4+0x0],xmm10
<+351>:	movups xmm13,XMMWORD PTR [r12+rax*4+0x10]
<+357>:	rcpps  xmm14,xmm13
<+361>:	mulps  xmm13,xmm14
<+365>:	mulps  xmm13,xmm14
<+369>:	addps  xmm14,xmm14
<+373>:	subps  xmm14,xmm13
<+377>:	mulps  xmm14,xmm2
<+381>:	movups XMMWORD PTR [r13+rax*4+0x10],xmm14
<+499>:	movups xmm9,XMMWORD PTR [r12+rax*4]
<+504>:	rcpps  xmm10,xmm9
<+508>:	mulps  xmm9,xmm10
<+512>:	mulps  xmm9,xmm10
<+516>:	addps  xmm10,xmm10
<+520>:	subps  xmm10,xmm9
<+524>:	mulps  xmm10,xmm2
<+528>:	movups XMMWORD PTR [r13+rax*4+0x0],xmm10
<+564>:	movups xmm13,XMMWORD PTR [r12+rax*4+0x10]
<+570>:	rcpps  xmm14,xmm13
<+574>:	mulps  xmm13,xmm14
<+578>:	mulps  xmm13,xmm14
<+582>:	addps  xmm14,xmm14
<+586>:	subps  xmm14,xmm13
<+590>:	mulps  xmm14,xmm2
<+594>:	movups XMMWORD PTR [r13+rax*4+0x10],xmm14
<+699>:	lea    rsi,[r13+rbx*4+0x0]
<+704>:	lea    rdi,[r12+rbx*4]
<+781>:	movups xmm9,XMMWORD PTR [rdi+rbx*4]
<+786>:	rcpps  xmm10,xmm9
<+790>:	mulps  xmm9,xmm10
<+794>:	mulps  xmm9,xmm10
<+798>:	addps  xmm10,xmm10
<+802>:	subps  xmm10,xmm9
<+806>:	mulps  xmm10,xmm2
<+810>:	movups XMMWORD PTR [rsi+rbx*4],xmm10
<+871>:	lea    r8,[r13+rbp*4+0x0]
<+876>:	lea    rdi,[r12+rbp*4]
<+894>:	movaps xmm1,xmm6
<+908>:	divss  xmm1,DWORD PTR [rdi+rcx*4]
<+913>:	movss  DWORD PTR [r8+rcx*4],xmm1

results->b[i] = arr->b[i] / c;
<+35>:	mov    r11,QWORD PTR [rsi+0x10]
<+46>:	mov    r10,QWORD PTR [rdi+0x10]
<+164>:	movss  xmm2,DWORD PTR [r10+rcx*4]
<+170>:	divss  xmm2,xmm7
<+174>:	movss  DWORD PTR [r11+rcx*4],xmm2
<+321>:	movups xmm11,XMMWORD PTR [r10+rax*4]
<+326>:	mulps  xmm11,xmm0
<+330>:	movups XMMWORD PTR [r11+rax*4],xmm11
<+387>:	movups xmm15,XMMWORD PTR [r10+rax*4+0x10]
<+393>:	mulps  xmm15,xmm0
<+397>:	movups XMMWORD PTR [r11+rax*4+0x10],xmm15
<+534>:	movups xmm11,XMMWORD PTR [r10+rax*4]
<+539>:	mulps  xmm11,xmm0
<+543>:	movups XMMWORD PTR [r11+rax*4],xmm11
<+600>:	movups xmm15,XMMWORD PTR [r10+rax*4+0x10]
<+606>:	mulps  xmm15,xmm0
<+610>:	movups XMMWORD PTR [r11+rax*4+0x10],xmm15
<+708>:	lea    rcx,[r11+rbx*4]
<+712>:	lea    r8,[r10+rbx*4]
<+729>:	rcpps  xmm12,xmm3
<+736>:	movaps xmm11,xmm3
<+747>:	mulps  xmm11,xmm12
<+757>:	mulps  xmm11,xmm12
<+761>:	addps  xmm12,xmm12
<+768>:	subps  xmm12,xmm11
<+815>:	movups xmm13,XMMWORD PTR [r8+rbx*4]
<+820>:	mulps  xmm13,xmm12
<+824>:	movups XMMWORD PTR [rcx+rbx*4],xmm13
<+880>:	lea    rsi,[r11+rbp*4]
<+884>:	lea    rbp,[r10+rbp*4]
<+919>:	movss  xmm2,DWORD PTR [rbp+rcx*4+0x0]
<+925>:	divss  xmm2,xmm7
<+929>:	movss  DWORD PTR [rsi+rcx*4],xmm2


44	}
<+942>:	pop    rbp
<+943>:	pop    rbx
<+944>:	pop    r15
<+946>:	pop    r14
<+948>:	pop    r13
<+950>:	pop    r12
<+952>:	ret    


