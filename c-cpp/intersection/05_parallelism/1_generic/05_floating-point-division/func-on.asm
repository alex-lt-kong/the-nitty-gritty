oid func_floating_division(float a, float b, float c, float d, float* arr, float* results, size_t arr_len) {
0x0000000000000000 <+0>:	movaps xmm5,xmm0
0x0000000000000003 <+3>:	mov    rcx,rdx

 for (int i = 0; i < arr_len; ++i) {
0x0000000000000006 <+6>:	test   rdx,rdx
0x0000000000000009 <+9>:	je     0x216 <func_floating_division+534>
0x000000000000000f <+15>:	lea    rax,[rdi+0xf]
0x0000000000000013 <+19>:	sub    rax,rsi
0x0000000000000016 <+22>:	cmp    rax,0x1e
0x000000000000001a <+26>:	jbe    0x1c0 <func_floating_division+448>
0x0000000000000020 <+32>:	lea    rax,[rdx-0x1]
0x0000000000000024 <+36>:	cmp    rax,0x2
0x0000000000000028 <+40>:	jbe    0x1c0 <func_floating_division+448>
0x000000000000002e <+46>:	mov    r8,rcx
0x0000000000000031 <+49>:	movaps xmm9,xmm0
0x0000000000000035 <+53>:	movaps xmm8,xmm1
0x0000000000000039 <+57>:	mov    rax,rdi
0x000000000000003c <+60>:	shr    r8,0x2
0x0000000000000040 <+64>:	movaps xmm7,xmm2
0x0000000000000043 <+67>:	movaps xmm6,xmm3
0x0000000000000046 <+70>:	mov    rdx,rsi
0x0000000000000049 <+73>:	shl    r8,0x4
0x000000000000004d <+77>:	shufps xmm9,xmm9,0x0
0x0000000000000052 <+82>:	shufps xmm7,xmm7,0x0
0x0000000000000056 <+86>:	shufps xmm6,xmm6,0x0
0x000000000000005a <+90>:	add    r8,rdi
0x000000000000005d <+93>:	shufps xmm8,xmm8,0x0
0x0000000000000062 <+98>:	nop    WORD PTR [rax+rax*1+0x0]
0x00000000000000b0 <+176>:	cmp    rax,r8
0x00000000000000b3 <+179>:	jne    0x68 <func_floating_division+104>
0x00000000000000b5 <+181>:	mov    rax,rcx
0x00000000000000b8 <+184>:	and    rax,0xfffffffffffffffc
0x00000000000000bc <+188>:	test   cl,0x3
0x00000000000000bf <+191>:	je     0x216 <func_floating_division+534>
0x0000000000000112 <+274>:	lea    edx,[rax+0x1]
0x0000000000000115 <+277>:	movsxd rdx,edx
0x0000000000000118 <+280>:	cmp    rcx,rdx
0x000000000000011b <+283>:	jbe    0x216 <func_floating_division+534>
0x0000000000000128 <+296>:	add    eax,0x2
0x0000000000000132 <+306>:	cdqe   
0x000000000000016f <+367>:	cmp    rcx,rax
0x0000000000000172 <+370>:	jbe    0x216 <func_floating_division+534>
0x00000000000001b6 <+438>:	ret    
0x00000000000001b7 <+439>:	nop    WORD PTR [rax+rax*1+0x0]
0x00000000000001c0 <+448>:	lea    rax,[rdi+rcx*4]
0x00000000000001c4 <+452>:	nop    DWORD PTR [rax+0x0]
0x00000000000001cf <+463>:	add    rdi,0x4
0x00000000000001d3 <+467>:	add    rsi,0x4
0x0000000000000211 <+529>:	cmp    rdi,rax
0x0000000000000214 <+532>:	jne    0x1c8 <func_floating_division+456>

   results[i] =  arr[i] / a;
0x0000000000000068 <+104>:	movups xmm4,XMMWORD PTR [rax]
0x0000000000000077 <+119>:	divps  xmm4,xmm9
0x000000000000007b <+123>:	movups XMMWORD PTR [rdx-0x10],xmm4
0x00000000000000c5 <+197>:	lea    rdx,[rax*4+0x0]
0x00000000000000d0 <+208>:	lea    r8,[rdi+rdx*1]
0x00000000000000d4 <+212>:	add    rdx,rsi
0x00000000000000d7 <+215>:	movss  xmm4,DWORD PTR [r8]
0x00000000000000dc <+220>:	divss  xmm4,xmm5
0x00000000000000e0 <+224>:	movss  DWORD PTR [rdx],xmm4
0x0000000000000121 <+289>:	shl    rdx,0x2
0x000000000000012b <+299>:	lea    r8,[rdi+rdx*1]
0x000000000000012f <+303>:	add    rdx,rsi
0x0000000000000134 <+308>:	movss  xmm4,DWORD PTR [r8]
0x0000000000000139 <+313>:	divss  xmm4,xmm5
0x000000000000013d <+317>:	movss  DWORD PTR [rdx],xmm4
0x0000000000000178 <+376>:	shl    rax,0x2
0x000000000000017c <+380>:	add    rdi,rax
0x000000000000017f <+383>:	add    rsi,rax
0x0000000000000182 <+386>:	movss  xmm0,DWORD PTR [rdi]
0x0000000000000186 <+390>:	divss  xmm0,xmm5
0x000000000000018a <+394>:	movss  DWORD PTR [rsi],xmm0
0x00000000000001c8 <+456>:	movss  xmm4,DWORD PTR [rdi]
0x00000000000001d7 <+471>:	divss  xmm4,xmm5
0x00000000000001db <+475>:	movss  DWORD PTR [rsi-0x4],xmm4

   results[i] += b / arr[i];
0x000000000000006b <+107>:	movaps xmm0,xmm8
0x000000000000006f <+111>:	add    rax,0x10
0x0000000000000073 <+115>:	add    rdx,0x10
0x000000000000007f <+127>:	movups xmm10,XMMWORD PTR [rax-0x10]
0x0000000000000084 <+132>:	divps  xmm0,xmm10
0x0000000000000088 <+136>:	addps  xmm4,xmm0
0x000000000000008b <+139>:	movups XMMWORD PTR [rdx-0x10],xmm4
0x00000000000000cd <+205>:	movaps xmm0,xmm1
0x00000000000000e4 <+228>:	divss  xmm0,DWORD PTR [r8]
0x00000000000000e9 <+233>:	addss  xmm4,xmm0
0x00000000000000ed <+237>:	movss  DWORD PTR [rdx],xmm4
0x0000000000000125 <+293>:	movaps xmm0,xmm1
0x0000000000000141 <+321>:	divss  xmm0,DWORD PTR [r8]
0x0000000000000146 <+326>:	addss  xmm4,xmm0
0x000000000000014a <+330>:	movss  DWORD PTR [rdx],xmm4
0x000000000000018e <+398>:	divss  xmm1,DWORD PTR [rdi]
0x0000000000000192 <+402>:	addss  xmm0,xmm1
0x0000000000000196 <+406>:	movss  DWORD PTR [rsi],xmm0
0x00000000000001cc <+460>:	movaps xmm0,xmm1
0x00000000000001e0 <+480>:	divss  xmm0,DWORD PTR [rdi-0x4]
0x00000000000001e5 <+485>:	addss  xmm4,xmm0
0x00000000000001e9 <+489>:	movss  DWORD PTR [rsi-0x4],xmm4

   results[i] += arr[i] / c;
0x000000000000008f <+143>:	movups xmm0,XMMWORD PTR [rax-0x10]
0x0000000000000093 <+147>:	divps  xmm0,xmm7
0x0000000000000096 <+150>:	addps  xmm0,xmm4
0x000000000000009c <+156>:	movups XMMWORD PTR [rdx-0x10],xmm0
0x00000000000000f1 <+241>:	movss  xmm0,DWORD PTR [r8]
0x00000000000000f6 <+246>:	divss  xmm0,xmm2
0x00000000000000fa <+250>:	addss  xmm0,xmm4
0x0000000000000101 <+257>:	movss  DWORD PTR [rdx],xmm0
0x000000000000014e <+334>:	movss  xmm0,DWORD PTR [r8]
0x0000000000000153 <+339>:	divss  xmm0,xmm2
0x0000000000000157 <+343>:	addss  xmm0,xmm4
0x000000000000015e <+350>:	movss  DWORD PTR [rdx],xmm0
0x000000000000019a <+410>:	movss  xmm1,DWORD PTR [rdi]
0x000000000000019e <+414>:	divss  xmm1,xmm2
0x00000000000001a2 <+418>:	addss  xmm1,xmm0
0x00000000000001a6 <+422>:	movss  DWORD PTR [rsi],xmm1
0x00000000000001ee <+494>:	movss  xmm0,DWORD PTR [rdi-0x4]
0x00000000000001f3 <+499>:	divss  xmm0,xmm2
0x00000000000001f7 <+503>:	addss  xmm0,xmm4
0x00000000000001fe <+510>:	movss  DWORD PTR [rsi-0x4],xmm0

   results[i] += d / arr[i];
0x0000000000000099 <+153>:	movaps xmm4,xmm6
0x00000000000000a0 <+160>:	movups xmm11,XMMWORD PTR [rax-0x10]
0x00000000000000a5 <+165>:	divps  xmm4,xmm11
0x00000000000000a9 <+169>:	addps  xmm0,xmm4
0x00000000000000ac <+172>:	movups XMMWORD PTR [rdx-0x10],xmm0
0x00000000000000fe <+254>:	movaps xmm4,xmm3
0x0000000000000105 <+261>:	divss  xmm4,DWORD PTR [r8]
0x000000000000010a <+266>:	addss  xmm0,xmm4
0x000000000000010e <+270>:	movss  DWORD PTR [rdx],xmm0
0x000000000000015b <+347>:	movaps xmm4,xmm3
0x0000000000000162 <+354>:	divss  xmm4,DWORD PTR [r8]
0x0000000000000167 <+359>:	addss  xmm0,xmm4
0x000000000000016b <+363>:	movss  DWORD PTR [rdx],xmm0
0x00000000000001aa <+426>:	divss  xmm3,DWORD PTR [rdi]
0x00000000000001ae <+430>:	addss  xmm3,xmm1
0x00000000000001b2 <+434>:	movss  DWORD PTR [rsi],xmm3
0x00000000000001fb <+507>:	movaps xmm4,xmm3
0x0000000000000203 <+515>:	divss  xmm4,DWORD PTR [rdi-0x4]
0x0000000000000208 <+520>:	addss  xmm0,xmm4
0x000000000000020c <+524>:	movss  DWORD PTR [rsi-0x4],xmm0

 }
}
0x0000000000000216 <+534>:	ret    
0x0000000000000217:	nop    WORD PTR [rax+rax*1+0x0]

 of assembler dump.
p of assembler code for function func_int_multiplication:
void func_int_multiplication(int32_t a, int32_t b, int32_t c, int32_t d, int32_t* arr, int32_t* results, size_t arr_len) {
0x0000000000000220 <+0>:	push   r12
0x0000000000000222 <+2>:	push   rbp
0x0000000000000223 <+3>:	push   rbx
0x0000000000000224 <+4>:	mov    rbx,QWORD PTR [rsp+0x20]

  for (int i = 0; i < arr_len; ++i) {
0x0000000000000229 <+9>:	test   rbx,rbx
0x000000000000022c <+12>:	je     0x46c <func_int_multiplication+588>
0x0000000000000232 <+18>:	lea    rax,[r8+0xf]
0x0000000000000236 <+22>:	mov    r10d,edi
0x0000000000000239 <+25>:	mov    edi,edx
0x000000000000023b <+27>:	sub    rax,r9
0x000000000000023e <+30>:	cmp    rax,0x1e
0x0000000000000242 <+34>:	jbe    0x478 <func_int_multiplication+600>
0x0000000000000248 <+40>:	lea    rax,[rbx-0x1]
0x000000000000024c <+44>:	cmp    rax,0x2
0x0000000000000250 <+48>:	jbe    0x478 <func_int_multiplication+600>
0x0000000000000256 <+54>:	movd   xmm2,r10d
0x000000000000025b <+59>:	movd   xmm3,esi
0x000000000000025f <+63>:	movd   xmm6,ecx
0x0000000000000263 <+67>:	mov    r11,rbx
0x0000000000000266 <+70>:	pshufd xmm5,xmm2,0x0
0x000000000000026b <+75>:	movd   xmm2,edx
0x000000000000026f <+79>:	pshufd xmm4,xmm3,0x0
0x0000000000000274 <+84>:	shr    r11,0x2
0x0000000000000278 <+88>:	pshufd xmm3,xmm2,0x0
0x000000000000027d <+93>:	pshufd xmm2,xmm6,0x0
0x0000000000000282 <+98>:	mov    rax,r8
0x0000000000000285 <+101>:	mov    rdx,r9
0x0000000000000288 <+104>:	shl    r11,0x4
0x000000000000028c <+108>:	movdqa xmm9,xmm5
0x0000000000000291 <+113>:	movdqa xmm8,xmm4
0x0000000000000296 <+118>:	movdqa xmm7,xmm3
0x000000000000029a <+122>:	movdqa xmm6,xmm2
0x000000000000029e <+126>:	add    r11,r8
0x00000000000002a1 <+129>:	psrlq  xmm9,0x20
0x00000000000002a7 <+135>:	psrlq  xmm8,0x20
0x00000000000002ad <+141>:	psrlq  xmm7,0x20
0x00000000000002b2 <+146>:	psrlq  xmm6,0x20
0x00000000000002b7 <+151>:	nop    WORD PTR [rax+rax*1+0x0]
0x000000000000038d <+365>:	cmp    rax,r11
0x0000000000000390 <+368>:	jne    0x2c0 <func_int_multiplication+160>
0x0000000000000396 <+374>:	mov    rax,rbx
0x0000000000000399 <+377>:	and    rax,0xfffffffffffffffc
0x000000000000039d <+381>:	test   bl,0x3
0x00000000000003a0 <+384>:	je     0x46c <func_int_multiplication+588>
0x00000000000003ea <+458>:	lea    edx,[rax+0x1]
0x00000000000003ed <+461>:	movsxd rdx,edx
0x00000000000003f0 <+464>:	cmp    rbx,rdx
0x00000000000003f3 <+467>:	jbe    0x46c <func_int_multiplication+588>
0x00000000000003f9 <+473>:	add    eax,0x2
0x0000000000000403 <+483>:	cdqe   
0x000000000000043a <+538>:	cmp    rbx,rax
0x000000000000043d <+541>:	jbe    0x46c <func_int_multiplication+588>
0x0000000000000483 <+611>:	add    r8,0x4
0x0000000000000487 <+615>:	add    r9,0x4
0x00000000000004ba <+666>:	cmp    r8,r11
0x00000000000004bd <+669>:	jne    0x480 <func_int_multiplication+608>

    results[i] =  arr[i] * a;
0x00000000000002c0 <+160>:	movdqu xmm10,XMMWORD PTR [rax]
0x00000000000002c5 <+165>:	movdqu xmm1,XMMWORD PTR [rax]
0x00000000000002c9 <+169>:	add    rax,0x10
0x00000000000002cd <+173>:	add    rdx,0x10
0x00000000000002d1 <+177>:	psrlq  xmm10,0x20
0x00000000000002d7 <+183>:	pmuludq xmm1,xmm5
0x00000000000002db <+187>:	pmuludq xmm10,xmm9
0x00000000000002e0 <+192>:	pshufd xmm1,xmm1,0x8
0x00000000000002e5 <+197>:	pshufd xmm10,xmm10,0x8
0x00000000000002eb <+203>:	punpckldq xmm1,xmm10
0x00000000000002f0 <+208>:	movups XMMWORD PTR [rdx-0x10],xmm1
0x00000000000003a6 <+390>:	lea    rdx,[rax*4+0x0]
0x00000000000003ae <+398>:	lea    rbp,[r8+rdx*1]
0x00000000000003b2 <+402>:	add    rdx,r9
0x00000000000003b5 <+405>:	mov    r12d,DWORD PTR [rbp+0x0]
0x00000000000003b9 <+409>:	imul   r12d,r10d
0x00000000000003bd <+413>:	mov    DWORD PTR [rdx],r12d
0x00000000000003f5 <+469>:	shl    rdx,0x2
0x00000000000003fc <+476>:	lea    rbp,[r8+rdx*1]
0x0000000000000400 <+480>:	add    rdx,r9
0x0000000000000405 <+485>:	mov    r12d,DWORD PTR [rbp+0x0]
0x0000000000000409 <+489>:	imul   r12d,r10d
0x000000000000040d <+493>:	mov    DWORD PTR [rdx],r12d
0x000000000000043f <+543>:	shl    rax,0x2
0x0000000000000443 <+547>:	add    r8,rax
0x0000000000000446 <+550>:	add    r9,rax
0x0000000000000449 <+553>:	imul   r10d,DWORD PTR [r8]
0x000000000000044d <+557>:	mov    DWORD PTR [r9],r10d
0x0000000000000480 <+608>:	mov    edx,DWORD PTR [r8]
0x000000000000048b <+619>:	imul   edx,r10d
0x000000000000048f <+623>:	mov    DWORD PTR [r9-0x4],edx

    results[i] += arr[i] * b;
0x00000000000002f4 <+212>:	movdqu xmm11,XMMWORD PTR [rax-0x10]
0x00000000000002fa <+218>:	movdqu xmm10,XMMWORD PTR [rax-0x10]
0x0000000000000300 <+224>:	psrlq  xmm11,0x20
0x0000000000000306 <+230>:	pmuludq xmm10,xmm4
0x000000000000030b <+235>:	pmuludq xmm11,xmm8
0x0000000000000310 <+240>:	pshufd xmm0,xmm10,0x8
0x0000000000000316 <+246>:	pshufd xmm10,xmm11,0x8
0x000000000000031c <+252>:	punpckldq xmm0,xmm10
0x0000000000000321 <+257>:	paddd  xmm1,xmm0
0x0000000000000325 <+261>:	movups XMMWORD PTR [rdx-0x10],xmm1
0x00000000000003c0 <+416>:	mov    r11d,DWORD PTR [rbp+0x0]
0x00000000000003c4 <+420>:	imul   r11d,esi
0x00000000000003c8 <+424>:	add    r12d,r11d
0x00000000000003cb <+427>:	mov    DWORD PTR [rdx],r12d
0x0000000000000410 <+496>:	mov    r11d,DWORD PTR [rbp+0x0]
0x0000000000000414 <+500>:	imul   r11d,esi
0x0000000000000418 <+504>:	add    r12d,r11d
0x000000000000041b <+507>:	mov    DWORD PTR [rdx],r12d
0x0000000000000450 <+560>:	imul   esi,DWORD PTR [r8]
0x0000000000000454 <+564>:	add    esi,r10d
0x0000000000000457 <+567>:	mov    DWORD PTR [r9],esi
0x0000000000000493 <+627>:	mov    eax,DWORD PTR [r8-0x4]
0x0000000000000497 <+631>:	imul   eax,esi
0x000000000000049a <+634>:	add    edx,eax
0x000000000000049c <+636>:	mov    DWORD PTR [r9-0x4],edx

    results[i] += arr[i] * c;
0x0000000000000329 <+265>:	movdqu xmm10,XMMWORD PTR [rax-0x10]
0x000000000000032f <+271>:	movdqu xmm0,XMMWORD PTR [rax-0x10]
0x0000000000000334 <+276>:	psrlq  xmm10,0x20
0x000000000000033a <+282>:	pmuludq xmm0,xmm3
0x000000000000033e <+286>:	pmuludq xmm10,xmm7
0x0000000000000343 <+291>:	pshufd xmm0,xmm0,0x8
0x0000000000000348 <+296>:	pshufd xmm10,xmm10,0x8
0x000000000000034e <+302>:	punpckldq xmm0,xmm10
0x0000000000000353 <+307>:	paddd  xmm0,xmm1
0x0000000000000357 <+311>:	movups XMMWORD PTR [rdx-0x10],xmm0
0x00000000000003ce <+430>:	mov    r11d,DWORD PTR [rbp+0x0]
0x00000000000003d2 <+434>:	imul   r11d,edi
0x00000000000003d6 <+438>:	add    r11d,r12d
0x00000000000003d9 <+441>:	mov    DWORD PTR [rdx],r11d
0x000000000000041e <+510>:	mov    r11d,DWORD PTR [rbp+0x0]
0x0000000000000422 <+514>:	imul   r11d,edi
0x0000000000000426 <+518>:	add    r11d,r12d
0x0000000000000429 <+521>:	mov    DWORD PTR [rdx],r11d
0x000000000000045a <+570>:	imul   edi,DWORD PTR [r8]
0x000000000000045e <+574>:	add    edi,esi
0x0000000000000460 <+576>:	mov    DWORD PTR [r9],edi
0x00000000000004a0 <+640>:	mov    eax,DWORD PTR [r8-0x4]
0x00000000000004a4 <+644>:	imul   eax,edi
0x00000000000004a7 <+647>:	add    eax,edx
0x00000000000004a9 <+649>:	mov    DWORD PTR [r9-0x4],eax

    results[i] += arr[i] * d;
0x000000000000035b <+315>:	movdqu xmm1,XMMWORD PTR [rax-0x10]
0x0000000000000360 <+320>:	movdqu xmm10,XMMWORD PTR [rax-0x10]
0x0000000000000366 <+326>:	psrlq  xmm1,0x20
0x000000000000036b <+331>:	pmuludq xmm10,xmm2
0x0000000000000370 <+336>:	pmuludq xmm1,xmm6
0x0000000000000374 <+340>:	pshufd xmm10,xmm10,0x8
0x000000000000037a <+346>:	pshufd xmm1,xmm1,0x8
0x000000000000037f <+351>:	punpckldq xmm10,xmm1
0x0000000000000384 <+356>:	paddd  xmm0,xmm10
0x0000000000000389 <+361>:	movups XMMWORD PTR [rdx-0x10],xmm0
0x00000000000003dc <+444>:	mov    r12d,DWORD PTR [rbp+0x0]
0x00000000000003e0 <+448>:	imul   r12d,ecx
0x00000000000003e4 <+452>:	add    r11d,r12d
0x00000000000003e7 <+455>:	mov    DWORD PTR [rdx],r11d
0x000000000000042c <+524>:	mov    r12d,DWORD PTR [rbp+0x0]
0x0000000000000430 <+528>:	imul   r12d,ecx
0x0000000000000434 <+532>:	add    r11d,r12d
0x0000000000000437 <+535>:	mov    DWORD PTR [rdx],r11d
0x0000000000000463 <+579>:	imul   ecx,DWORD PTR [r8]
0x0000000000000467 <+583>:	add    edi,ecx
0x0000000000000469 <+585>:	mov    DWORD PTR [r9],edi
0x00000000000004ad <+653>:	mov    edx,DWORD PTR [r8-0x4]
0x00000000000004b1 <+657>:	imul   edx,ecx
0x00000000000004b4 <+660>:	add    eax,edx
0x00000000000004b6 <+662>:	mov    DWORD PTR [r9-0x4],eax

  }
}
0x000000000000046c <+588>:	pop    rbx
0x000000000000046d <+589>:	pop    rbp
0x000000000000046e <+590>:	pop    r12
0x0000000000000470 <+592>:	ret    
0x0000000000000471 <+593>:	nop    DWORD PTR [rax+0x0]
0x0000000000000478 <+600>:	lea    r11,[r8+rbx*4]
0x000000000000047c <+604>:	nop    DWORD PTR [rax+0x0]
0x00000000000004bf <+671>:	pop    rbx
0x00000000000004c0 <+672>:	pop    rbp
0x00000000000004c1 <+673>:	pop    r12
0x00000000000004c3 <+675>:	ret    

