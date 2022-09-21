Dump of assembler code for function main:
   0x00000000000010e0 <+0>:	push   r15
   0x00000000000010e2 <+2>:	xor    edi,edi
   0x00000000000010e4 <+4>:	push   r14
   0x00000000000010e6 <+6>:	push   r13
   0x00000000000010e8 <+8>:	push   r12
   0x00000000000010ea <+10>:	push   rbp
   0x00000000000010eb <+11>:	push   rbx
   0x00000000000010ec <+12>:	sub    rsp,0x18
   0x00000000000010f0 <+16>:	call   0x1080 <time@plt>
   0x00000000000010f5 <+21>:	mov    rdi,rax
   0x00000000000010f8 <+24>:	call   0x1060 <srand@plt>
   0x00000000000010fd <+29>:	lea    rsi,[rip+0xf00]        # 0x2004
   0x0000000000001104 <+36>:	lea    rdi,[rip+0xefb]        # 0x2006
   0x000000000000110b <+43>:	call   0x10a0 <fopen@plt>
   0x0000000000001110 <+48>:	mov    QWORD PTR [rsp+0x8],rax
   0x0000000000001115 <+53>:	test   rax,rax
   0x0000000000001118 <+56>:	je     0x12f7 <main+535>
   0x000000000000111e <+62>:	mov    rcx,QWORD PTR [rsp+0x8]
   0x0000000000001123 <+67>:	mov    edx,0xe
   0x0000000000001128 <+72>:	mov    esi,0x1
   0x000000000000112d <+77>:	lea    rdi,[rip+0xef7]        # 0x202b
   0x0000000000001134 <+84>:	mov    ebp,0x1
   0x0000000000001139 <+89>:	mov    r12d,0x20
   0x000000000000113f <+95>:	call   0x10b0 <fwrite@plt>
   0x0000000000001144 <+100>:	mov    edi,0x40000000
   0x0000000000001149 <+105>:	call   0x1090 <malloc@plt>
   0x000000000000114e <+110>:	mov    r13,rax
   0x0000000000001151 <+113>:	mov    rbx,rax
   0x0000000000001154 <+116>:	lea    r15,[rax+0x40000000]
   0x000000000000115b <+123>:	mov    r14,rax
   0x000000000000115e <+126>:	xchg   ax,ax
   0x0000000000001160 <+128>:	call   0x10c0 <rand@plt>
   0x0000000000001165 <+133>:	add    r14,0x4
   0x0000000000001169 <+137>:	mov    DWORD PTR [r14-0x4],eax
   0x000000000000116d <+141>:	cmp    r14,r15
   0x0000000000001170 <+144>:	jne    0x1160 <main+128>
   0x0000000000001172 <+146>:	call   0x1040 <clock@plt>
   0x0000000000001177 <+151>:	mov    r15,rax
   0x000000000000117a <+154>:	cmp    ebp,0x1
   0x000000000000117d <+157>:	jne    0x123d <main+349>
   0x0000000000001183 <+163>:	mov    ecx,0x1
   0x0000000000001188 <+168>:	xor    eax,eax
   0x000000000000118a <+170>:	nop    WORD PTR [rax+rax*1+0x0]
   0x0000000000001190 <+176>:	lea    edx,[rax+rax*1]
   0x0000000000001193 <+179>:	add    DWORD PTR [r13+rax*4+0x0],edx
   0x0000000000001198 <+184>:	add    rax,0x1
   0x000000000000119c <+188>:	cmp    rax,0x10000000
   0x00000000000011a2 <+194>:	jne    0x1190 <main+176>
   0x00000000000011a4 <+196>:	lea    esi,[rcx+0x1]
   0x00000000000011a7 <+199>:	add    ecx,0x2
   0x00000000000011aa <+202>:	cmp    ecx,0x1f
   0x00000000000011ad <+205>:	jne    0x1188 <main+168>
   0x00000000000011af <+207>:	xor    ecx,ecx
   0x00000000000011b1 <+209>:	nop    DWORD PTR [rax+0x0]
   0x00000000000011b8 <+216>:	mov    eax,r12d
   0x00000000000011bb <+219>:	mov    edx,DWORD PTR [rbx]
   0x00000000000011bd <+221>:	sub    eax,esi
   0x00000000000011bf <+223>:	nop
   0x00000000000011c0 <+224>:	add    edx,ecx
   0x00000000000011c2 <+226>:	sub    eax,0x1
   0x00000000000011c5 <+229>:	jne    0x11c0 <main+224>
   0x00000000000011c7 <+231>:	add    ecx,0x1
   0x00000000000011ca <+234>:	mov    DWORD PTR [rbx],edx
   0x00000000000011cc <+236>:	add    rbx,0x4
   0x00000000000011d0 <+240>:	cmp    ecx,0x10000000
   0x00000000000011d6 <+246>:	jne    0x11b8 <main+216>
   0x00000000000011d8 <+248>:	call   0x1040 <clock@plt>
   0x00000000000011dd <+253>:	sub    rax,r15
   0x00000000000011e0 <+256>:	mov    rbx,rax
   0x00000000000011e3 <+259>:	call   0x10c0 <rand@plt>
   0x00000000000011e8 <+264>:	mov    rdi,QWORD PTR [rsp+0x8]
   0x00000000000011ed <+269>:	pxor   xmm0,xmm0
   0x00000000000011f1 <+273>:	mov    edx,0x1
   0x00000000000011f6 <+278>:	and    eax,0xfffffff
   0x00000000000011fb <+283>:	test   rbx,rbx
   0x00000000000011fe <+286>:	lea    rsi,[rip+0xe35]        # 0x203a
   0x0000000000001205 <+293>:	mov    ecx,DWORD PTR [r13+rax*4+0x0]
   0x000000000000120a <+298>:	lea    rax,[rbx+0x1f]
   0x000000000000120e <+302>:	cmovns rax,rbx
   0x0000000000001212 <+306>:	sar    rax,0x5
   0x0000000000001216 <+310>:	cvtsi2sd xmm0,rax
   0x000000000000121b <+315>:	mov    eax,0x1
   0x0000000000001220 <+320>:	divsd  xmm0,QWORD PTR [rip+0xe20]        # 0x2048
   0x0000000000001228 <+328>:	call   0x1070 <fprintf@plt>
   0x000000000000122d <+333>:	mov    rdi,r13
   0x0000000000001230 <+336>:	call   0x1030 <free@plt>
   0x0000000000001235 <+341>:	add    ebp,0x1
   0x0000000000001238 <+344>:	jmp    0x1144 <main+100>
   0x000000000000123d <+349>:	movsxd rcx,ebp
   0x0000000000001240 <+352>:	mov    esi,0x20
   0x0000000000001245 <+357>:	shl    rcx,0x2
   0x0000000000001249 <+361>:	nop    DWORD PTR [rax+0x0]
   0x0000000000001250 <+368>:	mov    rdx,r13
   0x0000000000001253 <+371>:	xor    eax,eax
   0x0000000000001255 <+373>:	nop    DWORD PTR [rax]
   0x0000000000001258 <+376>:	add    DWORD PTR [rdx],eax
   0x000000000000125a <+378>:	add    eax,ebp
   0x000000000000125c <+380>:	add    rdx,rcx
   0x000000000000125f <+383>:	cmp    eax,0xfffffff
   0x0000000000001264 <+388>:	jle    0x1258 <main+376>
   0x0000000000001266 <+390>:	sub    esi,0x1
   0x0000000000001269 <+393>:	jne    0x1250 <main+368>
   0x000000000000126b <+395>:	call   0x1040 <clock@plt>
   0x0000000000001270 <+400>:	sub    rax,r15
   0x0000000000001273 <+403>:	mov    rbx,rax
   0x0000000000001276 <+406>:	call   0x10c0 <rand@plt>
   0x000000000000127b <+411>:	mov    rdi,QWORD PTR [rsp+0x8]
   0x0000000000001280 <+416>:	pxor   xmm0,xmm0
   0x0000000000001284 <+420>:	mov    edx,ebp
   0x0000000000001286 <+422>:	and    eax,0xfffffff
   0x000000000000128b <+427>:	test   rbx,rbx
   0x000000000000128e <+430>:	lea    rsi,[rip+0xda5]        # 0x203a
   0x0000000000001295 <+437>:	mov    ecx,DWORD PTR [r13+rax*4+0x0]
   0x000000000000129a <+442>:	lea    rax,[rbx+0x1f]
   0x000000000000129e <+446>:	cmovns rax,rbx
   0x00000000000012a2 <+450>:	sar    rax,0x5
   0x00000000000012a6 <+454>:	cvtsi2sd xmm0,rax
   0x00000000000012ab <+459>:	mov    eax,0x1
   0x00000000000012b0 <+464>:	divsd  xmm0,QWORD PTR [rip+0xd90]        # 0x2048
   0x00000000000012b8 <+472>:	call   0x1070 <fprintf@plt>
   0x00000000000012bd <+477>:	mov    rdi,r13
   0x00000000000012c0 <+480>:	call   0x1030 <free@plt>
   0x00000000000012c5 <+485>:	cmp    ebp,0xf
   0x00000000000012c8 <+488>:	jle    0x1235 <main+341>
   0x00000000000012ce <+494>:	add    ebp,ebp
   0x00000000000012d0 <+496>:	cmp    ebp,0x1000
   0x00000000000012d6 <+502>:	jle    0x1144 <main+100>
   0x00000000000012dc <+508>:	mov    rdi,QWORD PTR [rsp+0x8]
   0x00000000000012e1 <+513>:	call   0x1050 <fclose@plt>
   0x00000000000012e6 <+518>:	add    rsp,0x18
   0x00000000000012ea <+522>:	xor    eax,eax
   0x00000000000012ec <+524>:	pop    rbx
   0x00000000000012ed <+525>:	pop    rbp
   0x00000000000012ee <+526>:	pop    r12
   0x00000000000012f0 <+528>:	pop    r13
   0x00000000000012f2 <+530>:	pop    r14
   0x00000000000012f4 <+532>:	pop    r15
   0x00000000000012f6 <+534>:	ret    
   0x00000000000012f7 <+535>:	mov    rcx,QWORD PTR [rip+0x2d82]        # 0x4080 <stderr@GLIBC_2.2.5>
   0x00000000000012fe <+542>:	mov    edx,0x14
   0x0000000000001303 <+547>:	mov    esi,0x1
   0x0000000000001308 <+552>:	lea    rdi,[rip+0xd07]        # 0x2016
   0x000000000000130f <+559>:	call   0x10b0 <fwrite@plt>
   0x0000000000001314 <+564>:	jmp    0x111e <main+62>
End of assembler dump.
