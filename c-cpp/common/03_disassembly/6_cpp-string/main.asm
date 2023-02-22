
main.out:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .text:

0000000000001140 <main>:
    1140:	41 55                	push   r13
    1142:	be 0c 00 00 00       	mov    esi,0xc
    1147:	41 54                	push   r12
    1149:	55                   	push   rbp
    114a:	53                   	push   rbx
    114b:	48 81 ec 88 00 00 00 	sub    rsp,0x88
    1152:	49 89 e5             	mov    r13,rsp
    1155:	4c 89 ef             	mov    rdi,r13
    1158:	e8 63 03 00 00       	call   14c0 <_ZNSt7__cxx119to_stringEi>
    115d:	48 8d 35 a0 0e 00 00 	lea    rsi,[rip+0xea0]        # 2004 <_IO_stdin_used+0x4>
    1164:	48 8d 3d 55 2f 00 00 	lea    rdi,[rip+0x2f55]        # 40c0 <_ZSt4cout@@GLIBCXX_3.4>
    116b:	e8 00 ff ff ff       	call   1070 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    1170:	48 8b 54 24 08       	mov    rdx,QWORD PTR [rsp+0x8]
    1175:	48 8b 34 24          	mov    rsi,QWORD PTR [rsp]
    1179:	48 89 c7             	mov    rdi,rax
    117c:	e8 ff fe ff ff       	call   1080 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
    1181:	48 89 c7             	mov    rdi,rax
    1184:	e8 a7 02 00 00       	call   1430 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0>
    1189:	4c 8d 64 24 60       	lea    r12,[rsp+0x60]
    118e:	be 38 00 00 00       	mov    esi,0x38
    1193:	4c 89 e7             	mov    rdi,r12
    1196:	e8 25 03 00 00       	call   14c0 <_ZNSt7__cxx119to_stringEi>
    119b:	48 8d 6c 24 40       	lea    rbp,[rsp+0x40]
    11a0:	be 22 00 00 00       	mov    esi,0x22
    11a5:	48 89 ef             	mov    rdi,rbp
    11a8:	e8 13 03 00 00       	call   14c0 <_ZNSt7__cxx119to_stringEi>
    11ad:	4c 8b 4c 24 40       	mov    r9,QWORD PTR [rsp+0x40]
    11b2:	48 8d 74 24 50       	lea    rsi,[rsp+0x50]
    11b7:	4c 8b 44 24 48       	mov    r8,QWORD PTR [rsp+0x48]
    11bc:	b8 0f 00 00 00       	mov    eax,0xf
    11c1:	48 8b 54 24 68       	mov    rdx,QWORD PTR [rsp+0x68]
    11c6:	48 89 c7             	mov    rdi,rax
    11c9:	49 39 f1             	cmp    r9,rsi
    11cc:	48 0f 45 7c 24 50    	cmovne rdi,QWORD PTR [rsp+0x50]
    11d2:	48 8b 74 24 60       	mov    rsi,QWORD PTR [rsp+0x60]
    11d7:	49 8d 0c 10          	lea    rcx,[r8+rdx*1]
    11db:	48 39 f9             	cmp    rcx,rdi
    11de:	76 17                	jbe    11f7 <main+0xb7>
    11e0:	48 8d 7c 24 70       	lea    rdi,[rsp+0x70]
    11e5:	48 39 fe             	cmp    rsi,rdi
    11e8:	48 0f 45 44 24 70    	cmovne rax,QWORD PTR [rsp+0x70]
    11ee:	48 39 c1             	cmp    rcx,rax
    11f1:	0f 86 a7 00 00 00    	jbe    129e <main+0x15e>
    11f7:	48 89 ef             	mov    rdi,rbp
    11fa:	e8 41 fe ff ff       	call   1040 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_appendEPKcm@plt>
    11ff:	48 8d 54 24 30       	lea    rdx,[rsp+0x30]
    1204:	48 89 54 24 20       	mov    QWORD PTR [rsp+0x20],rdx
    1209:	48 8b 08             	mov    rcx,QWORD PTR [rax]
    120c:	48 8d 50 10          	lea    rdx,[rax+0x10]
    1210:	48 39 d1             	cmp    rcx,rdx
    1213:	0f 84 bd 00 00 00    	je     12d6 <main+0x196>
    1219:	48 89 4c 24 20       	mov    QWORD PTR [rsp+0x20],rcx
    121e:	48 8b 48 10          	mov    rcx,QWORD PTR [rax+0x10]
    1222:	48 89 4c 24 30       	mov    QWORD PTR [rsp+0x30],rcx
    1227:	48 8b 48 08          	mov    rcx,QWORD PTR [rax+0x8]
    122b:	48 89 ef             	mov    rdi,rbp
    122e:	48 89 4c 24 28       	mov    QWORD PTR [rsp+0x28],rcx
    1233:	48 89 10             	mov    QWORD PTR [rax],rdx
    1236:	48 c7 40 08 00 00 00 	mov    QWORD PTR [rax+0x8],0x0
    123d:	00 
    123e:	c6 40 10 00          	mov    BYTE PTR [rax+0x10],0x0
    1242:	e8 69 fe ff ff       	call   10b0 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv@plt>
    1247:	4c 89 e7             	mov    rdi,r12
    124a:	e8 61 fe ff ff       	call   10b0 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv@plt>
    124f:	48 8d 35 b3 0d 00 00 	lea    rsi,[rip+0xdb3]        # 2009 <_IO_stdin_used+0x9>
    1256:	48 8d 3d 63 2e 00 00 	lea    rdi,[rip+0x2e63]        # 40c0 <_ZSt4cout@@GLIBCXX_3.4>
    125d:	e8 0e fe ff ff       	call   1070 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    1262:	48 8b 54 24 28       	mov    rdx,QWORD PTR [rsp+0x28]
    1267:	48 8b 74 24 20       	mov    rsi,QWORD PTR [rsp+0x20]
    126c:	48 89 c7             	mov    rdi,rax
    126f:	e8 0c fe ff ff       	call   1080 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
    1274:	48 89 c7             	mov    rdi,rax
    1277:	e8 b4 01 00 00       	call   1430 <_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0>
    127c:	48 8d 7c 24 20       	lea    rdi,[rsp+0x20]
    1281:	e8 2a fe ff ff       	call   10b0 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv@plt>
    1286:	4c 89 ef             	mov    rdi,r13
    1289:	e8 22 fe ff ff       	call   10b0 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv@plt>
    128e:	48 81 c4 88 00 00 00 	add    rsp,0x88
    1295:	31 c0                	xor    eax,eax
    1297:	5b                   	pop    rbx
    1298:	5d                   	pop    rbp
    1299:	41 5c                	pop    r12
    129b:	41 5d                	pop    r13
    129d:	c3                   	ret    
    129e:	4c 89 c9             	mov    rcx,r9
    12a1:	31 d2                	xor    edx,edx
    12a3:	31 f6                	xor    esi,esi
    12a5:	4c 89 e7             	mov    rdi,r12
    12a8:	e8 33 fe ff ff       	call   10e0 <_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_replaceEmmPKcm@plt>
    12ad:	48 8d 54 24 30       	lea    rdx,[rsp+0x30]
    12b2:	48 89 54 24 20       	mov    QWORD PTR [rsp+0x20],rdx
    12b7:	48 8b 08             	mov    rcx,QWORD PTR [rax]
    12ba:	48 8d 50 10          	lea    rdx,[rax+0x10]
    12be:	48 39 d1             	cmp    rcx,rdx
    12c1:	0f 85 52 ff ff ff    	jne    1219 <main+0xd9>
    12c7:	f3 0f 6f 48 10       	movdqu xmm1,XMMWORD PTR [rax+0x10]
    12cc:	0f 29 4c 24 30       	movaps XMMWORD PTR [rsp+0x30],xmm1
    12d1:	e9 51 ff ff ff       	jmp    1227 <main+0xe7>
    12d6:	f3 0f 6f 40 10       	movdqu xmm0,XMMWORD PTR [rax+0x10]
    12db:	0f 29 44 24 30       	movaps XMMWORD PTR [rsp+0x30],xmm0
    12e0:	e9 42 ff ff ff       	jmp    1227 <main+0xe7>
    12e5:	48 89 c5             	mov    rbp,rax
    12e8:	e9 23 fe ff ff       	jmp    1110 <main.cold>
    12ed:	48 89 c5             	mov    rbp,rax
    12f0:	e9 40 fe ff ff       	jmp    1135 <main.cold+0x25>
    12f5:	48 89 c3             	mov    rbx,rax
    12f8:	e9 2d fe ff ff       	jmp    112a <main.cold+0x1a>
    12fd:	48 89 c5             	mov    rbp,rax
    1300:	e9 15 fe ff ff       	jmp    111a <main.cold+0xa>

Disassembly of section .fini:
