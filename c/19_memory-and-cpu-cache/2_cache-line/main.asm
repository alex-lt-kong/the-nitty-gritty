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
   0x0000000000001110 <+48>:	mov    r14,rax
   0x0000000000001113 <+51>:	test   rax,rax
   0x0000000000001116 <+54>:	je     0x12d4 <main+500>
   0x000000000000111c <+60>:	mov    rcx,r14
   0x000000000000111f <+63>:	mov    edx,0x17
   0x0000000000001124 <+68>:	mov    esi,0x1
   0x0000000000001129 <+73>:	mov    r12d,0x1
   0x000000000000112f <+79>:	lea    rdi,[rip+0xef1]        # 0x2027
   0x0000000000001136 <+86>:	call   0x10b0 <fwrite@plt>
   0x000000000000113b <+91>:	mov    edi,0x40000000
   0x0000000000001140 <+96>:	call   0x1090 <malloc@plt>
   0x0000000000001145 <+101>:	mov    r13,rax
   0x0000000000001148 <+104>:	mov    rbp,rax
   0x000000000000114b <+107>:	lea    rbx,[rax+0x40000000]
   0x0000000000001152 <+114>:	mov    r15,rax
   0x0000000000001155 <+117>:	nop    DWORD PTR [rax]
   0x0000000000001158 <+120>:	call   0x10c0 <rand@plt>
   0x000000000000115d <+125>:	add    r15,0x4
   0x0000000000001161 <+129>:	mov    edx,eax
   0x0000000000001163 <+131>:	cdqe   
   0x0000000000001165 <+133>:	imul   rax,rax,0xffffffff80000005
   0x000000000000116c <+140>:	mov    ecx,edx
   0x000000000000116e <+142>:	sar    ecx,0x1f
   0x0000000000001171 <+145>:	shr    rax,0x20
   0x0000000000001175 <+149>:	add    eax,edx
   0x0000000000001177 <+151>:	sar    eax,0x1d
   0x000000000000117a <+154>:	sub    eax,ecx
   0x000000000000117c <+156>:	imul   eax,eax,0x3ffffffe
   0x0000000000001182 <+162>:	sub    edx,eax
   0x0000000000001184 <+164>:	mov    DWORD PTR [r15-0x4],edx
   0x0000000000001188 <+168>:	cmp    r15,rbx
   0x000000000000118b <+171>:	jne    0x1158 <main+120>
   0x000000000000118d <+173>:	call   0x1040 <clock@plt>
   0x0000000000001192 <+178>:	mov    rsi,rax
   0x0000000000001195 <+181>:	cmp    r12d,0x1
   0x0000000000001199 <+185>:	jne    0x1220 <main+320>
   0x000000000000119f <+191>:	xor    r15d,r15d
   0x00000000000011a2 <+194>:	nop    WORD PTR [rax+rax*1+0x0]
   0x00000000000011a8 <+200>:	mov    edx,DWORD PTR [rbp+0x0]
   0x00000000000011ab <+203>:	mov    eax,0x20
   0x00000000000011b0 <+208>:	add    r15d,edx
   0x00000000000011b3 <+211>:	sub    eax,0x1
   0x00000000000011b6 <+214>:	jne    0x11b0 <main+208>
   0x00000000000011b8 <+216>:	add    rbp,0x4
   0x00000000000011bc <+220>:	cmp    rbp,rbx
   0x00000000000011bf <+223>:	jne    0x11a8 <main+200>
   0x00000000000011c1 <+225>:	mov    QWORD PTR [rsp+0x8],rsi
   0x00000000000011c6 <+230>:	call   0x1040 <clock@plt>
   0x00000000000011cb <+235>:	mov    rsi,QWORD PTR [rsp+0x8]
   0x00000000000011d0 <+240>:	mov    rdi,r14
   0x00000000000011d3 <+243>:	mov    ecx,r15d
   0x00000000000011d6 <+246>:	pxor   xmm0,xmm0
   0x00000000000011da <+250>:	sub    rax,rsi
   0x00000000000011dd <+253>:	lea    rsi,[rip+0xe5b]        # 0x203f
   0x00000000000011e4 <+260>:	mov    rdx,rax
   0x00000000000011e7 <+263>:	lea    rax,[rax+0x1f]
   0x00000000000011eb <+267>:	cmovns rax,rdx
   0x00000000000011ef <+271>:	mov    edx,0x1
   0x00000000000011f4 <+276>:	sar    rax,0x5
   0x00000000000011f8 <+280>:	cvtsi2sd xmm0,rax
   0x00000000000011fd <+285>:	mov    eax,0x1
   0x0000000000001202 <+290>:	divsd  xmm0,QWORD PTR [rip+0xe46]        # 0x2050
   0x000000000000120a <+298>:	call   0x1070 <fprintf@plt>
   0x000000000000120f <+303>:	mov    rdi,r13
   0x0000000000001212 <+306>:	call   0x1030 <free@plt>
   0x0000000000001217 <+311>:	add    r12d,0x1
   0x000000000000121b <+315>:	jmp    0x113b <main+91>
   0x0000000000001220 <+320>:	movsxd rcx,r12d
   0x0000000000001223 <+323>:	mov    edi,0x20
   0x0000000000001228 <+328>:	xor    r15d,r15d
   0x000000000000122b <+331>:	shl    rcx,0x2
   0x000000000000122f <+335>:	nop
   0x0000000000001230 <+336>:	mov    rdx,r13
   0x0000000000001233 <+339>:	xor    eax,eax
   0x0000000000001235 <+341>:	nop    DWORD PTR [rax]
   0x0000000000001238 <+344>:	add    eax,r12d
   0x000000000000123b <+347>:	add    r15d,DWORD PTR [rdx]
   0x000000000000123e <+350>:	add    rdx,rcx
   0x0000000000001241 <+353>:	cmp    eax,0xfffffff
   0x0000000000001246 <+358>:	jle    0x1238 <main+344>
   0x0000000000001248 <+360>:	sub    edi,0x1
   0x000000000000124b <+363>:	jne    0x1230 <main+336>
   0x000000000000124d <+365>:	mov    QWORD PTR [rsp+0x8],rsi
   0x0000000000001252 <+370>:	call   0x1040 <clock@plt>
   0x0000000000001257 <+375>:	mov    rsi,QWORD PTR [rsp+0x8]
   0x000000000000125c <+380>:	mov    rdi,r14
   0x000000000000125f <+383>:	mov    ecx,r15d
   0x0000000000001262 <+386>:	pxor   xmm0,xmm0
   0x0000000000001266 <+390>:	sub    rax,rsi
   0x0000000000001269 <+393>:	lea    rsi,[rip+0xdcf]        # 0x203f
   0x0000000000001270 <+400>:	mov    rdx,rax
   0x0000000000001273 <+403>:	lea    rax,[rax+0x1f]
   0x0000000000001277 <+407>:	cmovns rax,rdx
   0x000000000000127b <+411>:	mov    edx,r12d
   0x000000000000127e <+414>:	sar    rax,0x5
   0x0000000000001282 <+418>:	cvtsi2sd xmm0,rax
   0x0000000000001287 <+423>:	mov    eax,0x1
   0x000000000000128c <+428>:	divsd  xmm0,QWORD PTR [rip+0xdbc]        # 0x2050
   0x0000000000001294 <+436>:	call   0x1070 <fprintf@plt>
   0x0000000000001299 <+441>:	mov    rdi,r13
   0x000000000000129c <+444>:	call   0x1030 <free@plt>
   0x00000000000012a1 <+449>:	cmp    r12d,0xf
   0x00000000000012a5 <+453>:	jle    0x1217 <main+311>
   0x00000000000012ab <+459>:	add    r12d,r12d
   0x00000000000012ae <+462>:	cmp    r12d,0x10000
   0x00000000000012b5 <+469>:	jle    0x113b <main+91>
   0x00000000000012bb <+475>:	mov    rdi,r14
   0x00000000000012be <+478>:	call   0x1050 <fclose@plt>
   0x00000000000012c3 <+483>:	add    rsp,0x18
   0x00000000000012c7 <+487>:	xor    eax,eax
   0x00000000000012c9 <+489>:	pop    rbx
   0x00000000000012ca <+490>:	pop    rbp
   0x00000000000012cb <+491>:	pop    r12
   0x00000000000012cd <+493>:	pop    r13
   0x00000000000012cf <+495>:	pop    r14
   0x00000000000012d1 <+497>:	pop    r15
   0x00000000000012d3 <+499>:	ret    
   0x00000000000012d4 <+500>:	mov    rcx,QWORD PTR [rip+0x2da5]        # 0x4080 <stderr@GLIBC_2.2.5>
   0x00000000000012db <+507>:	mov    edx,0x14
   0x00000000000012e0 <+512>:	mov    esi,0x1
   0x00000000000012e5 <+517>:	lea    rdi,[rip+0xd26]        # 0x2012
   0x00000000000012ec <+524>:	call   0x10b0 <fwrite@plt>
   0x00000000000012f1 <+529>:	jmp    0x111c <main+60>
End of assembler dump.
