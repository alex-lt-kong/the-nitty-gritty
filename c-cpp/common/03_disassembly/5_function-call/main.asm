
./main.out:     file format elf64-x86-64


Disassembly of section .init:

0000000000001000 <_init>:
    1000:	sub    rsp,0x8
    1004:	mov    rax,QWORD PTR [rip+0x2fdd]        # 3fe8 <__gmon_start__>
    100b:	test   rax,rax
    100e:	je     1012 <_init+0x12>
    1010:	call   rax
    1012:	add    rsp,0x8
    1016:	ret    

Disassembly of section .plt:

0000000000001020 <.plt>:
    1020:	push   QWORD PTR [rip+0x2fe2]        # 4008 <_GLOBAL_OFFSET_TABLE_+0x8>
    1026:	jmp    QWORD PTR [rip+0x2fe4]        # 4010 <_GLOBAL_OFFSET_TABLE_+0x10>
    102c:	nop    DWORD PTR [rax+0x0]

0000000000001030 <printf@plt>:
    1030:	jmp    QWORD PTR [rip+0x2fe2]        # 4018 <printf@GLIBC_2.2.5>
    1036:	push   0x0
    103b:	jmp    1020 <.plt>

Disassembly of section .plt.got:

0000000000001040 <__cxa_finalize@plt>:
    1040:	jmp    QWORD PTR [rip+0x2fb2]        # 3ff8 <__cxa_finalize@GLIBC_2.2.5>
    1046:	xchg   ax,ax

Disassembly of section .text:

0000000000001050 <_start>:
    1050:	xor    ebp,ebp
    1052:	mov    r9,rdx
    1055:	pop    rsi
    1056:	mov    rdx,rsp
    1059:	and    rsp,0xfffffffffffffff0
    105d:	push   rax
    105e:	push   rsp
    105f:	lea    r8,[rip+0x20a]        # 1270 <__libc_csu_fini>
    1066:	lea    rcx,[rip+0x1a3]        # 1210 <__libc_csu_init>
    106d:	lea    rdi,[rip+0x138]        # 11ac <main>
    1074:	call   QWORD PTR [rip+0x2f66]        # 3fe0 <__libc_start_main@GLIBC_2.2.5>
    107a:	hlt    
    107b:	nop    DWORD PTR [rax+rax*1+0x0]

0000000000001080 <deregister_tm_clones>:
    1080:	lea    rdi,[rip+0x2fa9]        # 4030 <__TMC_END__>
    1087:	lea    rax,[rip+0x2fa2]        # 4030 <__TMC_END__>
    108e:	cmp    rax,rdi
    1091:	je     10a8 <deregister_tm_clones+0x28>
    1093:	mov    rax,QWORD PTR [rip+0x2f3e]        # 3fd8 <_ITM_deregisterTMCloneTable>
    109a:	test   rax,rax
    109d:	je     10a8 <deregister_tm_clones+0x28>
    109f:	jmp    rax
    10a1:	nop    DWORD PTR [rax+0x0]
    10a8:	ret    
    10a9:	nop    DWORD PTR [rax+0x0]

00000000000010b0 <register_tm_clones>:
    10b0:	lea    rdi,[rip+0x2f79]        # 4030 <__TMC_END__>
    10b7:	lea    rsi,[rip+0x2f72]        # 4030 <__TMC_END__>
    10be:	sub    rsi,rdi
    10c1:	mov    rax,rsi
    10c4:	shr    rsi,0x3f
    10c8:	sar    rax,0x3
    10cc:	add    rsi,rax
    10cf:	sar    rsi,1
    10d2:	je     10e8 <register_tm_clones+0x38>
    10d4:	mov    rax,QWORD PTR [rip+0x2f15]        # 3ff0 <_ITM_registerTMCloneTable>
    10db:	test   rax,rax
    10de:	je     10e8 <register_tm_clones+0x38>
    10e0:	jmp    rax
    10e2:	nop    WORD PTR [rax+rax*1+0x0]
    10e8:	ret    
    10e9:	nop    DWORD PTR [rax+0x0]

00000000000010f0 <__do_global_dtors_aux>:
    10f0:	cmp    BYTE PTR [rip+0x2f39],0x0        # 4030 <__TMC_END__>
    10f7:	jne    1128 <__do_global_dtors_aux+0x38>
    10f9:	push   rbp
    10fa:	cmp    QWORD PTR [rip+0x2ef6],0x0        # 3ff8 <__cxa_finalize@GLIBC_2.2.5>
    1102:	mov    rbp,rsp
    1105:	je     1113 <__do_global_dtors_aux+0x23>
    1107:	mov    rdi,QWORD PTR [rip+0x2f1a]        # 4028 <__dso_handle>
    110e:	call   1040 <__cxa_finalize@plt>
    1113:	call   1080 <deregister_tm_clones>
    1118:	mov    BYTE PTR [rip+0x2f11],0x1        # 4030 <__TMC_END__>
    111f:	pop    rbp
    1120:	ret    
    1121:	nop    DWORD PTR [rax+0x0]
    1128:	ret    
    1129:	nop    DWORD PTR [rax+0x0]

0000000000001130 <frame_dummy>:
    1130:	jmp    10b0 <register_tm_clones>

0000000000001135 <add>:
    1135:	push   rbp
    1136:	mov    rbp,rsp
    1139:	mov    QWORD PTR [rbp-0x18],rdi
    113d:	mov    QWORD PTR [rbp-0x20],rsi
    1141:	mov    QWORD PTR [rbp-0x8],0x0
    1149:	mov    rdx,QWORD PTR [rbp-0x18]
    114d:	mov    rax,QWORD PTR [rbp-0x20]
    1151:	add    rax,rdx
    1154:	mov    QWORD PTR [rbp-0x8],rax
    1158:	mov    rax,QWORD PTR [rbp-0x8]
    115c:	pop    rbp
    115d:	ret    

000000000000115e <multiply>:
    115e:	push   rbp
    115f:	mov    rbp,rsp
    1162:	sub    rsp,0x20
    1166:	mov    QWORD PTR [rbp-0x18],rdi
    116a:	mov    QWORD PTR [rbp-0x20],rsi
    116e:	mov    QWORD PTR [rbp-0x8],0x0
    1176:	mov    QWORD PTR [rbp-0x10],0x0
    117e:	jmp    119c <multiply+0x3e>
    1180:	mov    rdx,QWORD PTR [rbp-0x18]
    1184:	mov    rax,QWORD PTR [rbp-0x8]
    1188:	mov    rsi,rdx
    118b:	mov    rdi,rax
    118e:	call   1135 <add>
    1193:	mov    QWORD PTR [rbp-0x8],rax
    1197:	add    QWORD PTR [rbp-0x10],0x1
    119c:	mov    rax,QWORD PTR [rbp-0x10]
    11a0:	cmp    rax,QWORD PTR [rbp-0x20]
    11a4:	jb     1180 <multiply+0x22>
    11a6:	mov    rax,QWORD PTR [rbp-0x8]
    11aa:	leave  
    11ab:	ret    

00000000000011ac <main>:
    11ac:	push   rbp
    11ad:	mov    rbp,rsp
    11b0:	sub    rsp,0x20
    11b4:	mov    QWORD PTR [rbp-0x8],0xc
    11bc:	mov    QWORD PTR [rbp-0x10],0x22
    11c4:	mov    rdx,QWORD PTR [rbp-0x10]
    11c8:	mov    rax,QWORD PTR [rbp-0x8]
    11cc:	mov    rsi,rdx
    11cf:	mov    rdi,rax
    11d2:	call   115e <multiply>
    11d7:	mov    QWORD PTR [rbp-0x18],rax
    11db:	mov    rcx,QWORD PTR [rbp-0x18]
    11df:	mov    rdx,QWORD PTR [rbp-0x10]
    11e3:	mov    rax,QWORD PTR [rbp-0x8]
    11e7:	mov    rsi,rax
    11ea:	lea    rdi,[rip+0xe13]        # 2004 <_IO_stdin_used+0x4>
    11f1:	mov    eax,0x0
    11f6:	call   1030 <printf@plt>
    11fb:	mov    eax,0x0
    1200:	leave  
    1201:	ret    
    1202:	nop    WORD PTR cs:[rax+rax*1+0x0]
    120c:	nop    DWORD PTR [rax+0x0]

0000000000001210 <__libc_csu_init>:
    1210:	push   r15
    1212:	lea    r15,[rip+0x2bcf]        # 3de8 <__frame_dummy_init_array_entry>
    1219:	push   r14
    121b:	mov    r14,rdx
    121e:	push   r13
    1220:	mov    r13,rsi
    1223:	push   r12
    1225:	mov    r12d,edi
    1228:	push   rbp
    1229:	lea    rbp,[rip+0x2bc0]        # 3df0 <__do_global_dtors_aux_fini_array_entry>
    1230:	push   rbx
    1231:	sub    rbp,r15
    1234:	sub    rsp,0x8
    1238:	call   1000 <_init>
    123d:	sar    rbp,0x3
    1241:	je     125e <__libc_csu_init+0x4e>
    1243:	xor    ebx,ebx
    1245:	nop    DWORD PTR [rax]
    1248:	mov    rdx,r14
    124b:	mov    rsi,r13
    124e:	mov    edi,r12d
    1251:	call   QWORD PTR [r15+rbx*8]
    1255:	add    rbx,0x1
    1259:	cmp    rbp,rbx
    125c:	jne    1248 <__libc_csu_init+0x38>
    125e:	add    rsp,0x8
    1262:	pop    rbx
    1263:	pop    rbp
    1264:	pop    r12
    1266:	pop    r13
    1268:	pop    r14
    126a:	pop    r15
    126c:	ret    
    126d:	nop    DWORD PTR [rax]

0000000000001270 <__libc_csu_fini>:
    1270:	ret    

Disassembly of section .fini:

0000000000001274 <_fini>:
    1274:	sub    rsp,0x8
    1278:	add    rsp,0x8
    127c:	ret    
