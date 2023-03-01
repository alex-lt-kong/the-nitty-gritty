
./main-o2.out:     file format elf64-x86-64


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

0000000000001050 <main>:
    1050:	sub    rsp,0x8
    1054:	mov    esi,0x22
    1059:	mov    edi,0xc
    105e:	call   1190 <multiply>
    1063:	mov    edx,0x22
    1068:	mov    esi,0xc
    106d:	lea    rdi,[rip+0xf90]        # 2004 <_IO_stdin_used+0x4>
    1074:	mov    rcx,rax
    1077:	xor    eax,eax
    1079:	call   1030 <printf@plt>
    107e:	xor    eax,eax
    1080:	add    rsp,0x8
    1084:	ret    
    1085:	nop    WORD PTR cs:[rax+rax*1+0x0]
    108f:	nop

0000000000001090 <_start>:
    1090:	xor    ebp,ebp
    1092:	mov    r9,rdx
    1095:	pop    rsi
    1096:	mov    rdx,rsp
    1099:	and    rsp,0xfffffffffffffff0
    109d:	push   rax
    109e:	push   rsp
    109f:	lea    r8,[rip+0x18a]        # 1230 <__libc_csu_fini>
    10a6:	lea    rcx,[rip+0x123]        # 11d0 <__libc_csu_init>
    10ad:	lea    rdi,[rip+0xffffffffffffff9c]        # 1050 <main>
    10b4:	call   QWORD PTR [rip+0x2f26]        # 3fe0 <__libc_start_main@GLIBC_2.2.5>
    10ba:	hlt    
    10bb:	nop    DWORD PTR [rax+rax*1+0x0]

00000000000010c0 <deregister_tm_clones>:
    10c0:	lea    rdi,[rip+0x2f69]        # 4030 <__TMC_END__>
    10c7:	lea    rax,[rip+0x2f62]        # 4030 <__TMC_END__>
    10ce:	cmp    rax,rdi
    10d1:	je     10e8 <deregister_tm_clones+0x28>
    10d3:	mov    rax,QWORD PTR [rip+0x2efe]        # 3fd8 <_ITM_deregisterTMCloneTable>
    10da:	test   rax,rax
    10dd:	je     10e8 <deregister_tm_clones+0x28>
    10df:	jmp    rax
    10e1:	nop    DWORD PTR [rax+0x0]
    10e8:	ret    
    10e9:	nop    DWORD PTR [rax+0x0]

00000000000010f0 <register_tm_clones>:
    10f0:	lea    rdi,[rip+0x2f39]        # 4030 <__TMC_END__>
    10f7:	lea    rsi,[rip+0x2f32]        # 4030 <__TMC_END__>
    10fe:	sub    rsi,rdi
    1101:	mov    rax,rsi
    1104:	shr    rsi,0x3f
    1108:	sar    rax,0x3
    110c:	add    rsi,rax
    110f:	sar    rsi,1
    1112:	je     1128 <register_tm_clones+0x38>
    1114:	mov    rax,QWORD PTR [rip+0x2ed5]        # 3ff0 <_ITM_registerTMCloneTable>
    111b:	test   rax,rax
    111e:	je     1128 <register_tm_clones+0x38>
    1120:	jmp    rax
    1122:	nop    WORD PTR [rax+rax*1+0x0]
    1128:	ret    
    1129:	nop    DWORD PTR [rax+0x0]

0000000000001130 <__do_global_dtors_aux>:
    1130:	cmp    BYTE PTR [rip+0x2ef9],0x0        # 4030 <__TMC_END__>
    1137:	jne    1168 <__do_global_dtors_aux+0x38>
    1139:	push   rbp
    113a:	cmp    QWORD PTR [rip+0x2eb6],0x0        # 3ff8 <__cxa_finalize@GLIBC_2.2.5>
    1142:	mov    rbp,rsp
    1145:	je     1153 <__do_global_dtors_aux+0x23>
    1147:	mov    rdi,QWORD PTR [rip+0x2eda]        # 4028 <__dso_handle>
    114e:	call   1040 <__cxa_finalize@plt>
    1153:	call   10c0 <deregister_tm_clones>
    1158:	mov    BYTE PTR [rip+0x2ed1],0x1        # 4030 <__TMC_END__>
    115f:	pop    rbp
    1160:	ret    
    1161:	nop    DWORD PTR [rax+0x0]
    1168:	ret    
    1169:	nop    DWORD PTR [rax+0x0]

0000000000001170 <frame_dummy>:
    1170:	jmp    10f0 <register_tm_clones>
    1175:	nop    WORD PTR cs:[rax+rax*1+0x0]
    117f:	nop

0000000000001180 <add>:
    1180:	lea    rax,[rdi+rsi*1]
    1184:	ret    
    1185:	data16 nop WORD PTR cs:[rax+rax*1+0x0]

0000000000001190 <multiply>:
    1190:	mov    r8,rdi
    1193:	mov    rcx,rsi
    1196:	test   rsi,rsi
    1199:	je     11c0 <multiply+0x30>
    119b:	xor    edx,edx
    119d:	xor    edi,edi
    119f:	nop
    11a0:	mov    rsi,r8
    11a3:	add    rdx,0x1
    11a7:	call   1180 <add>
    11ac:	mov    rdi,rax
    11af:	cmp    rcx,rdx
    11b2:	jne    11a0 <multiply+0x10>
    11b4:	mov    rax,rdi
    11b7:	ret    
    11b8:	nop    DWORD PTR [rax+rax*1+0x0]
    11c0:	xor    edi,edi
    11c2:	mov    rax,rdi
    11c5:	ret    
    11c6:	nop    WORD PTR cs:[rax+rax*1+0x0]

00000000000011d0 <__libc_csu_init>:
    11d0:	push   r15
    11d2:	lea    r15,[rip+0x2c0f]        # 3de8 <__frame_dummy_init_array_entry>
    11d9:	push   r14
    11db:	mov    r14,rdx
    11de:	push   r13
    11e0:	mov    r13,rsi
    11e3:	push   r12
    11e5:	mov    r12d,edi
    11e8:	push   rbp
    11e9:	lea    rbp,[rip+0x2c00]        # 3df0 <__do_global_dtors_aux_fini_array_entry>
    11f0:	push   rbx
    11f1:	sub    rbp,r15
    11f4:	sub    rsp,0x8
    11f8:	call   1000 <_init>
    11fd:	sar    rbp,0x3
    1201:	je     121e <__libc_csu_init+0x4e>
    1203:	xor    ebx,ebx
    1205:	nop    DWORD PTR [rax]
    1208:	mov    rdx,r14
    120b:	mov    rsi,r13
    120e:	mov    edi,r12d
    1211:	call   QWORD PTR [r15+rbx*8]
    1215:	add    rbx,0x1
    1219:	cmp    rbp,rbx
    121c:	jne    1208 <__libc_csu_init+0x38>
    121e:	add    rsp,0x8
    1222:	pop    rbx
    1223:	pop    rbp
    1224:	pop    r12
    1226:	pop    r13
    1228:	pop    r14
    122a:	pop    r15
    122c:	ret    
    122d:	nop    DWORD PTR [rax]

0000000000001230 <__libc_csu_fini>:
    1230:	ret    

Disassembly of section .fini:

0000000000001234 <_fini>:
    1234:	sub    rsp,0x8
    1238:	add    rsp,0x8
    123c:	ret    
