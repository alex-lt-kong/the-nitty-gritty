Dump of assembler code for function floating_division_aos:
21	void floating_division_aos(float a, float b, float c, struct pixel** arr, struct pixel** results, size_t arr_len) {
   0x0000000000000000 <+0>:	mov    r8,rsi
   0x0000000000000003 <+3>:	movaps xmm5,xmm2
   0x0000000000000006 <+6>:	movaps xmm4,xmm1
   0x0000000000000009 <+9>:	movaps xmm3,xmm0
   0x00000000000000b4 <+180>:	movaps xmm0,xmm3
   0x00000000000000b7 <+183>:	movaps xmm6,xmm5
   0x00000000000000ba <+186>:	shufps xmm0,xmm0,0x0
   0x00000000000000be <+190>:	movaps xmm2,xmm4
   0x00000000000000cd <+205>:	shufps xmm6,xmm6,0x0
   0x00000000000000e0 <+224>:	shufps xmm2,xmm2,0x0
   0x000000000000029a <+666>:	movaps xmm0,xmm3
   0x000000000000029d <+669>:	movaps xmm6,xmm5
   0x00000000000002a0 <+672>:	shufps xmm0,xmm0,0x0
   0x00000000000002a4 <+676>:	movaps xmm2,xmm4
   0x00000000000002b3 <+691>:	shufps xmm6,xmm6,0x0
   0x00000000000002c6 <+710>:	shufps xmm2,xmm2,0x0

22	  
23	  #if defined( __INTEL_COMPILER)
24	  #pragma ivdep
25	  // Pragmas are specific for the compiler and platform in use. So the best bet is to look at compiler's documentation.
26	  // https://stackoverflow.com/questions/5078679/what-is-the-scope-of-a-pragma-directive
27	  #elif defined(__GNUC__)
28	  #pragma GCC ivdep
29	  #endif
30	  for (int i = 0; i < arr_len; ++i) {
   0x000000000000000c <+12>:	test   rdx,rdx
   0x000000000000000f <+15>:	jbe    0x4c1 <floating_division_aos+1217>
   0x0000000000000015 <+21>:	cmp    rdx,0x4
   0x0000000000000019 <+25>:	jb     0x4c2 <floating_division_aos+1218>
   0x000000000000001f <+31>:	mov    rcx,rdi
   0x0000000000000022 <+34>:	and    rcx,0xf
   0x0000000000000026 <+38>:	je     0x3a <floating_division_aos+58>
   0x0000000000000028 <+40>:	test   rcx,0x7
   0x000000000000002f <+47>:	jne    0x4c2 <floating_division_aos+1218>
   0x0000000000000035 <+53>:	mov    ecx,0x1
   0x000000000000003a <+58>:	lea    rax,[rcx+0x4]
   0x000000000000003e <+62>:	cmp    rdx,rax
   0x0000000000000041 <+65>:	jb     0x4c2 <floating_division_aos+1218>
   0x0000000000000047 <+71>:	mov    rsi,rdx
   0x000000000000004a <+74>:	xor    r9d,r9d
   0x000000000000004d <+77>:	sub    rsi,rcx
   0x0000000000000050 <+80>:	xor    eax,eax
   0x0000000000000052 <+82>:	and    rsi,0x3
   0x0000000000000056 <+86>:	neg    rsi
   0x0000000000000059 <+89>:	add    rsi,rdx
   0x000000000000005c <+92>:	test   rcx,rcx
   0x000000000000005f <+95>:	jbe    0xa1 <floating_division_aos+161>
   0x000000000000006c <+108>:	inc    r9d
   0x000000000000006f <+111>:	inc    rax
   0x000000000000009c <+156>:	cmp    r9,rcx
   0x000000000000009f <+159>:	jb     0x61 <floating_division_aos+97>
   0x00000000000000a7 <+167>:	test   r9,0xf
   0x00000000000000ae <+174>:	je     0x29a <floating_division_aos+666>
   0x00000000000000eb <+235>:	add    eax,0x4
   0x0000000000000270 <+624>:	add    rcx,0x4
   0x000000000000028c <+652>:	cmp    rax,rsi
   0x000000000000028f <+655>:	jb     0xe7 <floating_division_aos+231>
   0x0000000000000295 <+661>:	jmp    0x479 <floating_division_aos+1145>
   0x00000000000002d1 <+721>:	add    eax,0x4
   0x0000000000000454 <+1108>:	add    rcx,0x4
   0x0000000000000470 <+1136>:	cmp    rax,rsi
   0x0000000000000473 <+1139>:	jb     0x2cd <floating_division_aos+717>
   0x0000000000000479 <+1145>:	movsxd rax,esi
   0x000000000000047c <+1148>:	mov    ecx,esi
   0x000000000000047e <+1150>:	mov    esi,esi
   0x0000000000000480 <+1152>:	cmp    rsi,rdx
   0x0000000000000483 <+1155>:	jae    0x4c1 <floating_division_aos+1217>
   0x0000000000000490 <+1168>:	inc    ecx
   0x0000000000000492 <+1170>:	inc    rax
   0x00000000000004bc <+1212>:	cmp    rcx,rdx
   0x00000000000004bf <+1215>:	jb     0x485 <floating_division_aos+1157>
   0x00000000000004c2 <+1218>:	xor    esi,esi
   0x00000000000004c4 <+1220>:	jmp    0x479 <floating_division_aos+1145>
   0x00000000000004c6 <+1222>:	nop    DWORD PTR [rax]
   0x00000000000004c9 <+1225>:	nop    DWORD PTR [rax+0x0]

31	    results[i]->r = arr[i]->r / a;
   0x0000000000000061 <+97>:	mov    r10,QWORD PTR [rdi+rax*8]
   0x0000000000000068 <+104>:	mov    r11,QWORD PTR [r8+rax*8]
   0x0000000000000072 <+114>:	movss  xmm0,DWORD PTR [r10]
   0x0000000000000083 <+131>:	divss  xmm0,xmm3
   0x000000000000008b <+139>:	movss  DWORD PTR [r11],xmm0
   0x00000000000000a1 <+161>:	mov    eax,ecx
   0x00000000000000c1 <+193>:	rcpps  xmm1,xmm0
   0x00000000000000c4 <+196>:	mulps  xmm0,xmm1
   0x00000000000000c7 <+199>:	mulps  xmm0,xmm1
   0x00000000000000ca <+202>:	addps  xmm1,xmm1
   0x00000000000000d1 <+209>:	subps  xmm1,xmm0
   0x00000000000000e7 <+231>:	mov    r9,QWORD PTR [rdi+rcx*8]
   0x00000000000000ee <+238>:	mov    r10,QWORD PTR [rdi+rcx*8+0x8]
   0x00000000000000f3 <+243>:	mov    r11,QWORD PTR [rdi+rcx*8+0x10]
   0x00000000000000f8 <+248>:	movd   xmm9,DWORD PTR [r9]
   0x00000000000000fd <+253>:	mov    r9,QWORD PTR [rdi+rcx*8+0x18]
   0x0000000000000102 <+258>:	movd   xmm6,DWORD PTR [r10]
   0x0000000000000107 <+263>:	movd   xmm8,DWORD PTR [r11]
   0x000000000000010c <+268>:	movd   xmm7,DWORD PTR [r9]
   0x0000000000000111 <+273>:	punpckldq xmm9,xmm6
   0x0000000000000116 <+278>:	punpckldq xmm8,xmm7
   0x000000000000011b <+283>:	movlhps xmm9,xmm8
   0x000000000000011f <+287>:	mulps  xmm9,xmm1
   0x0000000000000123 <+291>:	movdqu xmm11,XMMWORD PTR [r8+rcx*8+0x10]
   0x000000000000012a <+298>:	mov    r10,QWORD PTR [r8+rcx*8]
   0x000000000000012e <+302>:	movaps xmm12,xmm9
   0x0000000000000132 <+306>:	movq   r9,xmm11
   0x0000000000000137 <+311>:	punpckhqdq xmm11,xmm11
   0x000000000000013c <+316>:	movd   DWORD PTR [r10],xmm9
   0x0000000000000141 <+321>:	movq   r10,xmm11
   0x0000000000000146 <+326>:	movhlps xmm12,xmm9
   0x000000000000014a <+330>:	pshuflw xmm10,xmm9,0xee
   0x0000000000000150 <+336>:	mov    r11,QWORD PTR [r8+rcx*8+0x8]
   0x0000000000000155 <+341>:	pshuflw xmm13,xmm12,0xee
   0x000000000000015b <+347>:	movd   DWORD PTR [r11],xmm10
   0x0000000000000160 <+352>:	movd   DWORD PTR [r9],xmm12
   0x0000000000000165 <+357>:	movd   DWORD PTR [r10],xmm13
   0x00000000000002a7 <+679>:	rcpps  xmm1,xmm0
   0x00000000000002aa <+682>:	mulps  xmm0,xmm1
   0x00000000000002ad <+685>:	mulps  xmm0,xmm1
   0x00000000000002b0 <+688>:	addps  xmm1,xmm1
   0x00000000000002b7 <+695>:	subps  xmm1,xmm0
   0x00000000000002cd <+717>:	mov    r9,QWORD PTR [rdi+rcx*8]
   0x00000000000002d4 <+724>:	mov    r10,QWORD PTR [rdi+rcx*8+0x8]
   0x00000000000002d9 <+729>:	mov    r11,QWORD PTR [rdi+rcx*8+0x10]
   0x00000000000002de <+734>:	movd   xmm9,DWORD PTR [r9]
   0x00000000000002e3 <+739>:	mov    r9,QWORD PTR [rdi+rcx*8+0x18]
   0x00000000000002e8 <+744>:	movd   xmm6,DWORD PTR [r10]
   0x00000000000002ed <+749>:	movd   xmm8,DWORD PTR [r11]
   0x00000000000002f2 <+754>:	movd   xmm7,DWORD PTR [r9]
   0x00000000000002f7 <+759>:	punpckldq xmm9,xmm6
   0x00000000000002fc <+764>:	punpckldq xmm8,xmm7
   0x0000000000000301 <+769>:	movlhps xmm9,xmm8
   0x0000000000000305 <+773>:	mulps  xmm9,xmm1
   0x0000000000000309 <+777>:	mov    r10,QWORD PTR [r8+rcx*8]
   0x000000000000030d <+781>:	movaps xmm12,xmm9
   0x0000000000000311 <+785>:	movdqu xmm11,XMMWORD PTR [r8+rcx*8+0x10]
   0x0000000000000318 <+792>:	movq   r9,xmm11
   0x000000000000031d <+797>:	punpckhqdq xmm11,xmm11
   0x0000000000000322 <+802>:	movd   DWORD PTR [r10],xmm9
   0x0000000000000327 <+807>:	movq   r10,xmm11
   0x000000000000032c <+812>:	movhlps xmm12,xmm9
   0x0000000000000330 <+816>:	pshuflw xmm10,xmm9,0xee
   0x0000000000000336 <+822>:	mov    r11,QWORD PTR [r8+rcx*8+0x8]
   0x000000000000033b <+827>:	pshuflw xmm13,xmm12,0xee
   0x0000000000000341 <+833>:	movd   DWORD PTR [r11],xmm10
   0x0000000000000346 <+838>:	movd   DWORD PTR [r9],xmm12
   0x000000000000034b <+843>:	movd   DWORD PTR [r10],xmm13
   0x0000000000000485 <+1157>:	mov    rsi,QWORD PTR [rdi+rax*8]
   0x000000000000048c <+1164>:	mov    r9,QWORD PTR [r8+rax*8]
   0x0000000000000495 <+1173>:	movss  xmm0,DWORD PTR [rsi]
   0x00000000000004a3 <+1187>:	divss  xmm0,xmm3
   0x00000000000004ab <+1195>:	movss  DWORD PTR [r9],xmm0

32	    results[i]->g = b / arr[i]->g;
   0x0000000000000065 <+101>:	movaps xmm1,xmm4
   0x000000000000007d <+125>:	divss  xmm1,DWORD PTR [r10+0x4]
   0x0000000000000090 <+144>:	movss  DWORD PTR [r11+0x4],xmm1
   0x000000000000016a <+362>:	mov    r11,QWORD PTR [rdi+rcx*8]
   0x000000000000016e <+366>:	mov    r9,QWORD PTR [rdi+rcx*8+0x8]
   0x0000000000000173 <+371>:	mov    r10,QWORD PTR [rdi+rcx*8+0x10]
   0x0000000000000178 <+376>:	movd   xmm6,DWORD PTR [r11+0x4]
   0x000000000000017e <+382>:	mov    r11,QWORD PTR [rdi+rcx*8+0x18]
   0x0000000000000183 <+387>:	movd   xmm14,DWORD PTR [r9+0x4]
   0x0000000000000189 <+393>:	punpckldq xmm6,xmm14
   0x000000000000018e <+398>:	movd   xmm14,DWORD PTR [r10+0x4]
   0x0000000000000194 <+404>:	movd   xmm15,DWORD PTR [r11+0x4]
   0x000000000000019a <+410>:	punpckldq xmm14,xmm15
   0x000000000000019f <+415>:	movlhps xmm6,xmm14
   0x00000000000001a3 <+419>:	rcpps  xmm7,xmm6
   0x00000000000001a6 <+422>:	movdqu xmm9,XMMWORD PTR [r8+rcx*8+0x10]
   0x00000000000001ad <+429>:	mulps  xmm6,xmm7
   0x00000000000001b0 <+432>:	movq   r11,xmm9
   0x00000000000001b5 <+437>:	punpckhqdq xmm9,xmm9
   0x00000000000001ba <+442>:	mulps  xmm6,xmm7
   0x00000000000001bd <+445>:	addps  xmm7,xmm7
   0x00000000000001c0 <+448>:	mov    r9,QWORD PTR [r8+rcx*8]
   0x00000000000001c4 <+452>:	subps  xmm7,xmm6
   0x00000000000001c7 <+455>:	mulps  xmm7,xmm2
   0x00000000000001ca <+458>:	movd   DWORD PTR [r9+0x4],xmm7
   0x00000000000001d0 <+464>:	movaps xmm10,xmm7
   0x00000000000001d4 <+468>:	movq   r9,xmm9
   0x00000000000001d9 <+473>:	movhlps xmm10,xmm7
   0x00000000000001dd <+477>:	pshuflw xmm8,xmm7,0xee
   0x00000000000001e3 <+483>:	mov    r10,QWORD PTR [r8+rcx*8+0x8]
   0x00000000000001e8 <+488>:	pshuflw xmm11,xmm10,0xee
   0x00000000000001ee <+494>:	movd   DWORD PTR [r10+0x4],xmm8
   0x00000000000001f4 <+500>:	movd   DWORD PTR [r11+0x4],xmm10
   0x00000000000001fa <+506>:	movd   DWORD PTR [r9+0x4],xmm11
   0x0000000000000350 <+848>:	mov    r11,QWORD PTR [rdi+rcx*8]
   0x0000000000000354 <+852>:	mov    r9,QWORD PTR [rdi+rcx*8+0x8]
   0x0000000000000359 <+857>:	mov    r10,QWORD PTR [rdi+rcx*8+0x10]
   0x000000000000035e <+862>:	movd   xmm6,DWORD PTR [r11+0x4]
   0x0000000000000364 <+868>:	mov    r11,QWORD PTR [rdi+rcx*8+0x18]
   0x0000000000000369 <+873>:	movd   xmm14,DWORD PTR [r9+0x4]
   0x000000000000036f <+879>:	punpckldq xmm6,xmm14
   0x0000000000000374 <+884>:	movd   xmm14,DWORD PTR [r10+0x4]
   0x000000000000037a <+890>:	movd   xmm15,DWORD PTR [r11+0x4]
   0x0000000000000380 <+896>:	punpckldq xmm14,xmm15
   0x0000000000000385 <+901>:	movlhps xmm6,xmm14
   0x0000000000000389 <+905>:	rcpps  xmm7,xmm6
   0x000000000000038c <+908>:	mulps  xmm6,xmm7
   0x000000000000038f <+911>:	mulps  xmm6,xmm7
   0x0000000000000392 <+914>:	addps  xmm7,xmm7
   0x0000000000000395 <+917>:	mov    r9,QWORD PTR [r8+rcx*8]
   0x0000000000000399 <+921>:	subps  xmm7,xmm6
   0x000000000000039c <+924>:	mulps  xmm7,xmm2
   0x000000000000039f <+927>:	movdqu xmm9,XMMWORD PTR [r8+rcx*8+0x10]
   0x00000000000003a6 <+934>:	movaps xmm10,xmm7
   0x00000000000003aa <+938>:	movq   r11,xmm9
   0x00000000000003af <+943>:	punpckhqdq xmm9,xmm9
   0x00000000000003b4 <+948>:	movd   DWORD PTR [r9+0x4],xmm7
   0x00000000000003ba <+954>:	movq   r9,xmm9
   0x00000000000003bf <+959>:	movhlps xmm10,xmm7
   0x00000000000003c3 <+963>:	pshuflw xmm8,xmm7,0xee
   0x00000000000003c9 <+969>:	mov    r10,QWORD PTR [r8+rcx*8+0x8]
   0x00000000000003ce <+974>:	pshuflw xmm11,xmm10,0xee
   0x00000000000003d4 <+980>:	movd   DWORD PTR [r10+0x4],xmm8
   0x00000000000003da <+986>:	movd   DWORD PTR [r11+0x4],xmm10
   0x00000000000003e0 <+992>:	movd   DWORD PTR [r9+0x4],xmm11
   0x0000000000000489 <+1161>:	movaps xmm1,xmm4
   0x000000000000049e <+1182>:	divss  xmm1,DWORD PTR [rsi+0x4]
   0x00000000000004b0 <+1200>:	movss  DWORD PTR [r9+0x4],xmm1

33	    results[i]->b = arr[i]->b / c;
   0x0000000000000077 <+119>:	movss  xmm2,DWORD PTR [r10+0x8]
   0x0000000000000087 <+135>:	divss  xmm2,xmm5
   0x0000000000000096 <+150>:	movss  DWORD PTR [r11+0x8],xmm2
   0x00000000000000a3 <+163>:	lea    r9,[r8+rcx*8]
   0x00000000000000d4 <+212>:	rcpps  xmm0,xmm6
   0x00000000000000d7 <+215>:	mulps  xmm6,xmm0
   0x00000000000000da <+218>:	mulps  xmm6,xmm0
   0x00000000000000dd <+221>:	addps  xmm0,xmm0
   0x00000000000000e4 <+228>:	subps  xmm0,xmm6
   0x0000000000000200 <+512>:	mov    r10,QWORD PTR [rdi+rcx*8]
   0x0000000000000204 <+516>:	mov    r11,QWORD PTR [rdi+rcx*8+0x8]
   0x0000000000000209 <+521>:	mov    r9,QWORD PTR [rdi+rcx*8+0x10]
   0x000000000000020e <+526>:	movd   xmm6,DWORD PTR [r10+0x8]
   0x0000000000000214 <+532>:	mov    r10,QWORD PTR [rdi+rcx*8+0x18]
   0x0000000000000219 <+537>:	movd   xmm12,DWORD PTR [r11+0x8]
   0x000000000000021f <+543>:	punpckldq xmm6,xmm12
   0x0000000000000224 <+548>:	movd   xmm12,DWORD PTR [r9+0x8]
   0x000000000000022a <+554>:	xchg   ax,ax
   0x000000000000022c <+556>:	movd   xmm13,DWORD PTR [r10+0x8]
   0x0000000000000232 <+562>:	punpckldq xmm12,xmm13
   0x0000000000000237 <+567>:	movlhps xmm6,xmm12
   0x000000000000023b <+571>:	mulps  xmm6,xmm0
   0x000000000000023e <+574>:	movdqu xmm7,XMMWORD PTR [r8+rcx*8+0x10]
   0x0000000000000245 <+581>:	mov    r11,QWORD PTR [r8+rcx*8]
   0x0000000000000249 <+585>:	movaps xmm8,xmm6
   0x000000000000024d <+589>:	movq   r10,xmm7
   0x0000000000000252 <+594>:	punpckhqdq xmm7,xmm7
   0x0000000000000256 <+598>:	movd   DWORD PTR [r11+0x8],xmm6
   0x000000000000025c <+604>:	movq   r11,xmm7
   0x0000000000000261 <+609>:	movhlps xmm8,xmm6
   0x0000000000000265 <+613>:	pshuflw xmm15,xmm6,0xee
   0x000000000000026b <+619>:	mov    r9,QWORD PTR [r8+rcx*8+0x8]
   0x0000000000000274 <+628>:	pshuflw xmm6,xmm8,0xee
   0x000000000000027a <+634>:	movd   DWORD PTR [r9+0x8],xmm15
   0x0000000000000280 <+640>:	movd   DWORD PTR [r10+0x8],xmm8
   0x0000000000000286 <+646>:	movd   DWORD PTR [r11+0x8],xmm6
   0x00000000000002ba <+698>:	rcpps  xmm0,xmm6
   0x00000000000002bd <+701>:	mulps  xmm6,xmm0
   0x00000000000002c0 <+704>:	mulps  xmm6,xmm0
   0x00000000000002c3 <+707>:	addps  xmm0,xmm0
   0x00000000000002ca <+714>:	subps  xmm0,xmm6
   0x00000000000003e6 <+998>:	mov    r10,QWORD PTR [rdi+rcx*8]
   0x00000000000003ea <+1002>:	mov    r11,QWORD PTR [rdi+rcx*8+0x8]
   0x00000000000003ef <+1007>:	mov    r9,QWORD PTR [rdi+rcx*8+0x10]
   0x00000000000003f4 <+1012>:	movd   xmm6,DWORD PTR [r10+0x8]
   0x00000000000003fa <+1018>:	mov    r10,QWORD PTR [rdi+rcx*8+0x18]
   0x00000000000003ff <+1023>:	movd   xmm12,DWORD PTR [r11+0x8]
   0x0000000000000405 <+1029>:	punpckldq xmm6,xmm12
   0x000000000000040a <+1034>:	movd   xmm12,DWORD PTR [r9+0x8]
   0x0000000000000410 <+1040>:	movd   xmm13,DWORD PTR [r10+0x8]
   0x0000000000000416 <+1046>:	punpckldq xmm12,xmm13
   0x000000000000041b <+1051>:	movlhps xmm6,xmm12
   0x000000000000041f <+1055>:	mulps  xmm6,xmm0
   0x0000000000000422 <+1058>:	mov    r11,QWORD PTR [r8+rcx*8]
   0x0000000000000426 <+1062>:	movaps xmm8,xmm6
   0x000000000000042a <+1066>:	movdqu xmm7,XMMWORD PTR [r8+rcx*8+0x10]
   0x0000000000000431 <+1073>:	movq   r10,xmm7
   0x0000000000000436 <+1078>:	punpckhqdq xmm7,xmm7
   0x000000000000043a <+1082>:	movd   DWORD PTR [r11+0x8],xmm6
   0x0000000000000440 <+1088>:	movq   r11,xmm7
   0x0000000000000445 <+1093>:	movhlps xmm8,xmm6
   0x0000000000000449 <+1097>:	pshuflw xmm15,xmm6,0xee
   0x000000000000044f <+1103>:	mov    r9,QWORD PTR [r8+rcx*8+0x8]
   0x0000000000000458 <+1112>:	pshuflw xmm6,xmm8,0xee
   0x000000000000045e <+1118>:	movd   DWORD PTR [r9+0x8],xmm15
   0x0000000000000464 <+1124>:	movd   DWORD PTR [r10+0x8],xmm8
   0x000000000000046a <+1130>:	movd   DWORD PTR [r11+0x8],xmm6
   0x0000000000000499 <+1177>:	movss  xmm2,DWORD PTR [rsi+0x8]
   0x00000000000004a7 <+1191>:	divss  xmm2,xmm5
   0x00000000000004b6 <+1206>:	movss  DWORD PTR [r9+0x8],xmm2

34	  }
35	}
   0x00000000000004c1 <+1217>:	ret    

End of assembler dump.
Dump of assembler code for function floating_division_soa:
37	void floating_division_soa(float a, float b, float c, struct pixelArray* arr, struct pixelArray* results, size_t arr_len) {
   0x00000000000004d0 <+0>:	push   r12
   0x00000000000004d2 <+2>:	push   r13
   0x00000000000004d4 <+4>:	push   r14
   0x00000000000004d6 <+6>:	push   r15
   0x00000000000004d8 <+8>:	push   rbx
   0x00000000000004d9 <+9>:	push   rbp
   0x00000000000004da <+10>:	movaps xmm7,xmm2
   0x00000000000004dd <+13>:	movaps xmm6,xmm1
   0x00000000000004e0 <+16>:	movaps xmm5,xmm0
   0x00000000000005a0 <+208>:	movaps xmm4,xmm5
   0x00000000000005a3 <+211>:	movaps xmm3,xmm7
   0x00000000000005a6 <+214>:	shufps xmm4,xmm4,0x0
   0x00000000000005aa <+218>:	movaps xmm2,xmm6
   0x00000000000005ad <+221>:	rcpps  xmm1,xmm4
   0x00000000000005b0 <+224>:	movaps xmm0,xmm1
   0x00000000000005b3 <+227>:	mulps  xmm0,xmm4
   0x00000000000005b6 <+230>:	mulps  xmm0,xmm1
   0x00000000000005b9 <+233>:	addps  xmm1,xmm1
   0x00000000000005bc <+236>:	shufps xmm3,xmm3,0x0
   0x00000000000005c0 <+240>:	subps  xmm1,xmm0
   0x00000000000005c3 <+243>:	rcpps  xmm0,xmm3
   0x00000000000005c6 <+246>:	movaps xmm8,xmm0
   0x00000000000005ca <+250>:	mulps  xmm8,xmm3
   0x00000000000005ce <+254>:	mulps  xmm8,xmm0
   0x00000000000005d2 <+258>:	addps  xmm0,xmm0
   0x00000000000005d5 <+261>:	shufps xmm2,xmm2,0x0
   0x00000000000005d9 <+265>:	subps  xmm0,xmm8
   0x0000000000000675 <+421>:	movaps xmm4,xmm5
   0x0000000000000678 <+424>:	movaps xmm3,xmm7
   0x000000000000067b <+427>:	shufps xmm4,xmm4,0x0
   0x000000000000067f <+431>:	movaps xmm2,xmm6
   0x0000000000000682 <+434>:	rcpps  xmm1,xmm4
   0x0000000000000685 <+437>:	movaps xmm0,xmm1
   0x0000000000000688 <+440>:	mulps  xmm0,xmm4
   0x000000000000068b <+443>:	mulps  xmm0,xmm1
   0x000000000000068e <+446>:	addps  xmm1,xmm1
   0x0000000000000691 <+449>:	shufps xmm3,xmm3,0x0
   0x0000000000000695 <+453>:	subps  xmm1,xmm0
   0x0000000000000698 <+456>:	rcpps  xmm0,xmm3
   0x000000000000069b <+459>:	movaps xmm8,xmm0
   0x000000000000069f <+463>:	mulps  xmm8,xmm3
   0x00000000000006a3 <+467>:	mulps  xmm8,xmm0
   0x00000000000006a7 <+471>:	addps  xmm0,xmm0
   0x00000000000006aa <+474>:	shufps xmm2,xmm2,0x0
   0x00000000000006ae <+478>:	subps  xmm0,xmm8

38	  #if defined( __INTEL_COMPILER)
39	  #pragma ivdep
40	  #elif defined(__GNUC__)
41	  #pragma GCC ivdep
42	  #endif
43	  for (int i = 0; i < arr_len; ++i) {
   0x00000000000004e3 <+19>:	test   rdx,rdx
   0x00000000000004e6 <+22>:	jbe    0x87e <floating_division_soa+942>
   0x0000000000000502 <+50>:	cmp    rdx,0x8
   0x0000000000000506 <+54>:	jb     0x894 <floating_division_soa+964>
   0x000000000000050c <+60>:	mov    rax,r11
   0x000000000000050f <+63>:	and    rax,0xf
   0x0000000000000513 <+67>:	je     0x52c <floating_division_soa+92>
   0x0000000000000515 <+69>:	test   rax,0x3
   0x000000000000051b <+75>:	jne    0x88d <floating_division_soa+957>
   0x0000000000000521 <+81>:	neg    rax
   0x0000000000000524 <+84>:	add    rax,0x10
   0x0000000000000528 <+88>:	shr    rax,0x2
   0x000000000000052c <+92>:	lea    rcx,[rax+0x8]
   0x0000000000000530 <+96>:	cmp    rdx,rcx
   0x0000000000000533 <+99>:	jb     0x88d <floating_division_soa+957>
   0x0000000000000539 <+105>:	mov    r9,rdx
   0x000000000000053c <+108>:	xor    ebx,ebx
   0x000000000000053e <+110>:	sub    r9,rax
   0x0000000000000541 <+113>:	xor    ecx,ecx
   0x0000000000000543 <+115>:	and    r9,0x7
   0x0000000000000547 <+119>:	neg    r9
   0x000000000000054a <+122>:	add    r9,rdx
   0x000000000000054d <+125>:	test   rax,rax
   0x0000000000000550 <+128>:	jbe    0x58c <floating_division_soa+188>
   0x0000000000000565 <+149>:	inc    ebx
   0x0000000000000584 <+180>:	inc    rcx
   0x0000000000000587 <+183>:	cmp    rbx,rax
   0x000000000000058a <+186>:	jb     0x552 <floating_division_soa+130>
   0x0000000000000593 <+195>:	test   rbx,0xf
   0x000000000000059a <+202>:	je     0x675 <floating_division_soa+421>
   0x00000000000005e6 <+278>:	add    ecx,0x8
   0x0000000000000663 <+403>:	add    rax,0x8
   0x0000000000000667 <+407>:	cmp    rcx,r9
   0x000000000000066a <+410>:	jb     0x5dd <floating_division_soa+269>
   0x0000000000000670 <+416>:	jmp    0x745 <floating_division_soa+629>
   0x00000000000006bb <+491>:	add    ecx,0x8
   0x0000000000000738 <+616>:	add    rax,0x8
   0x000000000000073c <+620>:	cmp    rcx,r9
   0x000000000000073f <+623>:	jb     0x6b2 <floating_division_soa+482>
   0x0000000000000745 <+629>:	lea    rax,[r9+0x1]
   0x0000000000000749 <+633>:	cmp    rax,rdx
   0x000000000000074c <+636>:	ja     0x87e <floating_division_soa+942>
   0x0000000000000752 <+642>:	sub    rdx,r9
   0x0000000000000755 <+645>:	cmp    rdx,0x4
   0x0000000000000759 <+649>:	jb     0x889 <floating_division_soa+953>
   0x0000000000000762 <+658>:	mov    rax,rdx
   0x0000000000000765 <+661>:	mov    DWORD PTR [rsp-0x8],0x0
   0x000000000000076d <+669>:	and    rax,0xfffffffffffffffc
   0x0000000000000771 <+673>:	mov    QWORD PTR [rsp-0x18],r15
   0x0000000000000776 <+678>:	mov    QWORD PTR [rsp-0x10],rdx
   0x000000000000077f <+687>:	mov    rdx,rcx
   0x0000000000000786 <+694>:	mov    r15d,DWORD PTR [rsp-0x8]
   0x000000000000079c <+716>:	xor    ebx,ebx
   0x000000000000079e <+718>:	xchg   ax,ax
   0x00000000000007b7 <+743>:	add    r15d,0x4
   0x000000000000080d <+829>:	add    rbx,0x4
   0x0000000000000811 <+833>:	cmp    r15,rax
   0x0000000000000814 <+836>:	jb     0x7a0 <floating_division_soa+720>
   0x0000000000000816 <+838>:	mov    r15,QWORD PTR [rsp-0x18]
   0x000000000000081b <+843>:	mov    rdx,QWORD PTR [rsp-0x10]
   0x0000000000000820 <+848>:	movsxd rcx,eax
   0x0000000000000823 <+851>:	mov    ebx,eax
   0x0000000000000825 <+853>:	mov    eax,eax
   0x0000000000000827 <+855>:	cmp    rax,rdx
   0x000000000000082a <+858>:	jae    0x87e <floating_division_soa+942>
   0x000000000000085a <+906>:	inc    ebx
   0x0000000000000876 <+934>:	inc    rcx
   0x0000000000000879 <+937>:	cmp    rbx,rdx
   0x000000000000087c <+940>:	jb     0x848 <floating_division_soa+888>
   0x0000000000000889 <+953>:	xor    eax,eax
   0x000000000000088b <+955>:	jmp    0x820 <floating_division_soa+848>
   0x000000000000088d <+957>:	xor    r9d,r9d
   0x0000000000000890 <+960>:	xor    eax,eax
   0x0000000000000892 <+962>:	jmp    0x820 <floating_division_soa+848>
   0x0000000000000894 <+964>:	cmp    rdx,0x1
   0x0000000000000898 <+968>:	jae    0x88d <floating_division_soa+957>
   0x000000000000089a <+970>:	jmp    0x87e <floating_division_soa+942>
   0x000000000000089c <+972>:	nop    DWORD PTR [rax+0x0]

44	    results->r[i] = arr->r[i] / a;
   0x00000000000004ec <+28>:	mov    r15,QWORD PTR [rsi]
   0x00000000000004f7 <+39>:	mov    r14,QWORD PTR [rdi]
   0x0000000000000552 <+130>:	movss  xmm0,DWORD PTR [r14+rcx*4]
   0x000000000000055b <+139>:	divss  xmm0,xmm5
   0x000000000000055f <+143>:	movss  DWORD PTR [r15+rcx*4],xmm0
   0x000000000000058c <+188>:	mov    ecx,eax
   0x00000000000005dd <+269>:	movups xmm8,XMMWORD PTR [r14+rax*4]
   0x00000000000005e2 <+274>:	mulps  xmm8,xmm1
   0x00000000000005e9 <+281>:	movups XMMWORD PTR [r15+rax*4],xmm8
   0x000000000000061f <+335>:	movups xmm12,XMMWORD PTR [r14+rax*4+0x10]
   0x0000000000000625 <+341>:	mulps  xmm12,xmm1
   0x0000000000000629 <+345>:	movups XMMWORD PTR [r15+rax*4+0x10],xmm12
   0x00000000000006b2 <+482>:	movups xmm8,XMMWORD PTR [r14+rax*4]
   0x00000000000006b7 <+487>:	mulps  xmm8,xmm1
   0x00000000000006be <+494>:	movups XMMWORD PTR [r15+rax*4],xmm8
   0x00000000000006f4 <+548>:	movups xmm12,XMMWORD PTR [r14+rax*4+0x10]
   0x00000000000006fa <+554>:	mulps  xmm12,xmm1
   0x00000000000006fe <+558>:	movups XMMWORD PTR [r15+rax*4+0x10],xmm12
   0x000000000000075f <+655>:	movsxd rbx,r9d
   0x000000000000077b <+683>:	lea    rcx,[r15+rbx*4]
   0x0000000000000782 <+690>:	lea    rbp,[r14+rbx*4]
   0x00000000000007a0 <+720>:	rcpps  xmm1,xmm4
   0x00000000000007a3 <+723>:	movups xmm8,XMMWORD PTR [rbp+rbx*4+0x0]
   0x00000000000007ad <+733>:	movaps xmm0,xmm4
   0x00000000000007b4 <+740>:	mulps  xmm0,xmm1
   0x00000000000007bf <+751>:	mulps  xmm0,xmm1
   0x00000000000007c2 <+754>:	addps  xmm1,xmm1
   0x00000000000007cd <+765>:	subps  xmm1,xmm0
   0x00000000000007d4 <+772>:	mulps  xmm8,xmm1
   0x00000000000007d8 <+776>:	movups XMMWORD PTR [rdx+rbx*4],xmm8
   0x000000000000082c <+860>:	movsxd rbp,r9d
   0x000000000000082f <+863>:	lea    rax,[r15+rbp*4]
   0x0000000000000833 <+867>:	lea    r9,[r14+rbp*4]
   0x0000000000000848 <+888>:	movss  xmm0,DWORD PTR [r9+rcx*4]
   0x0000000000000851 <+897>:	divss  xmm0,xmm5
   0x0000000000000855 <+901>:	movss  DWORD PTR [rax+rcx*4],xmm0

45	    results->g[i] = b / arr->g[i];
   0x00000000000004ef <+31>:	mov    r13,QWORD PTR [rsi+0x8]
   0x00000000000004fa <+42>:	mov    r12,QWORD PTR [rdi+0x8]
   0x0000000000000558 <+136>:	movaps xmm1,xmm6
   0x0000000000000567 <+151>:	divss  xmm1,DWORD PTR [r12+rcx*4]
   0x000000000000056d <+157>:	movss  DWORD PTR [r13+rcx*4+0x0],xmm1
   0x000000000000058e <+190>:	lea    rbx,[r13+rax*4+0x0]
   0x00000000000005ee <+286>:	movups xmm9,XMMWORD PTR [r12+rax*4]
   0x00000000000005f3 <+291>:	rcpps  xmm10,xmm9
   0x00000000000005f7 <+295>:	mulps  xmm9,xmm10
   0x00000000000005fb <+299>:	mulps  xmm9,xmm10
   0x00000000000005ff <+303>:	addps  xmm10,xmm10
   0x0000000000000603 <+307>:	subps  xmm10,xmm9
   0x0000000000000607 <+311>:	mulps  xmm10,xmm2
   0x000000000000060b <+315>:	movups XMMWORD PTR [r13+rax*4+0x0],xmm10
   0x000000000000062f <+351>:	movups xmm13,XMMWORD PTR [r12+rax*4+0x10]
   0x0000000000000635 <+357>:	rcpps  xmm14,xmm13
   0x0000000000000639 <+361>:	mulps  xmm13,xmm14
   0x000000000000063d <+365>:	mulps  xmm13,xmm14
   0x0000000000000641 <+369>:	addps  xmm14,xmm14
   0x0000000000000645 <+373>:	subps  xmm14,xmm13
   0x0000000000000649 <+377>:	mulps  xmm14,xmm2
   0x000000000000064d <+381>:	movups XMMWORD PTR [r13+rax*4+0x10],xmm14
   0x00000000000006c3 <+499>:	movups xmm9,XMMWORD PTR [r12+rax*4]
   0x00000000000006c8 <+504>:	rcpps  xmm10,xmm9
   0x00000000000006cc <+508>:	mulps  xmm9,xmm10
   0x00000000000006d0 <+512>:	mulps  xmm9,xmm10
   0x00000000000006d4 <+516>:	addps  xmm10,xmm10
   0x00000000000006d8 <+520>:	subps  xmm10,xmm9
   0x00000000000006dc <+524>:	mulps  xmm10,xmm2
   0x00000000000006e0 <+528>:	movups XMMWORD PTR [r13+rax*4+0x0],xmm10
   0x0000000000000704 <+564>:	movups xmm13,XMMWORD PTR [r12+rax*4+0x10]
   0x000000000000070a <+570>:	rcpps  xmm14,xmm13
   0x000000000000070e <+574>:	mulps  xmm13,xmm14
   0x0000000000000712 <+578>:	mulps  xmm13,xmm14
   0x0000000000000716 <+582>:	addps  xmm14,xmm14
   0x000000000000071a <+586>:	subps  xmm14,xmm13
   0x000000000000071e <+590>:	mulps  xmm14,xmm2
   0x0000000000000722 <+594>:	movups XMMWORD PTR [r13+rax*4+0x10],xmm14
   0x000000000000078b <+699>:	lea    rsi,[r13+rbx*4+0x0]
   0x0000000000000790 <+704>:	lea    rdi,[r12+rbx*4]
   0x00000000000007dd <+781>:	movups xmm9,XMMWORD PTR [rdi+rbx*4]
   0x00000000000007e2 <+786>:	rcpps  xmm10,xmm9
   0x00000000000007e6 <+790>:	mulps  xmm9,xmm10
   0x00000000000007ea <+794>:	mulps  xmm9,xmm10
   0x00000000000007ee <+798>:	addps  xmm10,xmm10
   0x00000000000007f2 <+802>:	subps  xmm10,xmm9
   0x00000000000007f6 <+806>:	mulps  xmm10,xmm2
   0x00000000000007fa <+810>:	movups XMMWORD PTR [rsi+rbx*4],xmm10
   0x0000000000000837 <+871>:	lea    r8,[r13+rbp*4+0x0]
   0x000000000000083c <+876>:	lea    rdi,[r12+rbp*4]
   0x000000000000084e <+894>:	movaps xmm1,xmm6
   0x000000000000085c <+908>:	divss  xmm1,DWORD PTR [rdi+rcx*4]
   0x0000000000000861 <+913>:	movss  DWORD PTR [r8+rcx*4],xmm1

46	    results->b[i] = arr->b[i] / c;
   0x00000000000004f3 <+35>:	mov    r11,QWORD PTR [rsi+0x10]
   0x00000000000004fe <+46>:	mov    r10,QWORD PTR [rdi+0x10]
   0x0000000000000574 <+164>:	movss  xmm2,DWORD PTR [r10+rcx*4]
   0x000000000000057a <+170>:	divss  xmm2,xmm7
   0x000000000000057e <+174>:	movss  DWORD PTR [r11+rcx*4],xmm2
   0x0000000000000611 <+321>:	movups xmm11,XMMWORD PTR [r10+rax*4]
   0x0000000000000616 <+326>:	mulps  xmm11,xmm0
   0x000000000000061a <+330>:	movups XMMWORD PTR [r11+rax*4],xmm11
   0x0000000000000653 <+387>:	movups xmm15,XMMWORD PTR [r10+rax*4+0x10]
   0x0000000000000659 <+393>:	mulps  xmm15,xmm0
   0x000000000000065d <+397>:	movups XMMWORD PTR [r11+rax*4+0x10],xmm15
   0x00000000000006e6 <+534>:	movups xmm11,XMMWORD PTR [r10+rax*4]
   0x00000000000006eb <+539>:	mulps  xmm11,xmm0
   0x00000000000006ef <+543>:	movups XMMWORD PTR [r11+rax*4],xmm11
   0x0000000000000728 <+600>:	movups xmm15,XMMWORD PTR [r10+rax*4+0x10]
   0x000000000000072e <+606>:	mulps  xmm15,xmm0
   0x0000000000000732 <+610>:	movups XMMWORD PTR [r11+rax*4+0x10],xmm15
   0x0000000000000794 <+708>:	lea    rcx,[r11+rbx*4]
   0x0000000000000798 <+712>:	lea    r8,[r10+rbx*4]
   0x00000000000007a9 <+729>:	rcpps  xmm12,xmm3
   0x00000000000007b0 <+736>:	movaps xmm11,xmm3
   0x00000000000007bb <+747>:	mulps  xmm11,xmm12
   0x00000000000007c5 <+757>:	mulps  xmm11,xmm12
   0x00000000000007c9 <+761>:	addps  xmm12,xmm12
   0x00000000000007d0 <+768>:	subps  xmm12,xmm11
   0x00000000000007ff <+815>:	movups xmm13,XMMWORD PTR [r8+rbx*4]
   0x0000000000000804 <+820>:	mulps  xmm13,xmm12
   0x0000000000000808 <+824>:	movups XMMWORD PTR [rcx+rbx*4],xmm13
   0x0000000000000840 <+880>:	lea    rsi,[r11+rbp*4]
   0x0000000000000844 <+884>:	lea    rbp,[r10+rbp*4]
   0x0000000000000867 <+919>:	movss  xmm2,DWORD PTR [rbp+rcx*4+0x0]
   0x000000000000086d <+925>:	divss  xmm2,xmm7
   0x0000000000000871 <+929>:	movss  DWORD PTR [rsi+rcx*4],xmm2

47	  }
48	}
   0x000000000000087e <+942>:	pop    rbp
   0x000000000000087f <+943>:	pop    rbx
   0x0000000000000880 <+944>:	pop    r15
   0x0000000000000882 <+946>:	pop    r14
   0x0000000000000884 <+948>:	pop    r13
   0x0000000000000886 <+950>:	pop    r12
   0x0000000000000888 <+952>:	ret    

End of assembler dump.
