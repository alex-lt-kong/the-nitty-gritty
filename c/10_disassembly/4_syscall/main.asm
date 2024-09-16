
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

0000000000001030 <puts@plt>:
    1030:	jmp    QWORD PTR [rip+0x2fe2]        # 4018 <puts@GLIBC_2.2.5>
    1036:	push   0x0
    103b:	jmp    1020 <.plt>

0000000000001040 <close@plt>:
    1040:	jmp    QWORD PTR [rip+0x2fda]        # 4020 <close@GLIBC_2.2.5>
    1046:	push   0x1
    104b:	jmp    1020 <.plt>

0000000000001050 <read@plt>:
    1050:	jmp    QWORD PTR [rip+0x2fd2]        # 4028 <read@GLIBC_2.2.5>
    1056:	push   0x2
    105b:	jmp    1020 <.plt>

0000000000001060 <open@plt>:
    1060:	jmp    QWORD PTR [rip+0x2fca]        # 4030 <open@GLIBC_2.2.5>
    1066:	push   0x3
    106b:	jmp    1020 <.plt>

Disassembly of section .plt.got:

0000000000001070 <__cxa_finalize@plt>:
    1070:	jmp    QWORD PTR [rip+0x2f82]        # 3ff8 <__cxa_finalize@GLIBC_2.2.5>
    1076:	xchg   ax,ax

Disassembly of section .text:

0000000000001080 <_start>:
    1080:	xor    ebp,ebp
    1082:	mov    r9,rdx
    1085:	pop    rsi
    1086:	mov    rdx,rsp
    1089:	and    rsp,0xfffffffffffffff0
    108d:	push   rax
    108e:	push   rsp
    108f:	lea    r8,[rip+0x21a]        # 12b0 <__libc_csu_fini>
    1096:	lea    rcx,[rip+0x1b3]        # 1250 <__libc_csu_init>
    109d:	lea    rdi,[rip+0xc1]        # 1165 <main>
    10a4:	call   QWORD PTR [rip+0x2f36]        # 3fe0 <__libc_start_main@GLIBC_2.2.5>
    10aa:	hlt    
    10ab:	nop    DWORD PTR [rax+rax*1+0x0]

00000000000010b0 <deregister_tm_clones>:
    10b0:	lea    rdi,[rip+0x2f91]        # 4048 <__TMC_END__>
    10b7:	lea    rax,[rip+0x2f8a]        # 4048 <__TMC_END__>
    10be:	cmp    rax,rdi
    10c1:	je     10d8 <deregister_tm_clones+0x28>
    10c3:	mov    rax,QWORD PTR [rip+0x2f0e]        # 3fd8 <_ITM_deregisterTMCloneTable>
    10ca:	test   rax,rax
    10cd:	je     10d8 <deregister_tm_clones+0x28>
    10cf:	jmp    rax
    10d1:	nop    DWORD PTR [rax+0x0]
    10d8:	ret    
    10d9:	nop    DWORD PTR [rax+0x0]

00000000000010e0 <register_tm_clones>:
    10e0:	lea    rdi,[rip+0x2f61]        # 4048 <__TMC_END__>
    10e7:	lea    rsi,[rip+0x2f5a]        # 4048 <__TMC_END__>
    10ee:	sub    rsi,rdi
    10f1:	mov    rax,rsi
    10f4:	shr    rsi,0x3f
    10f8:	sar    rax,0x3
    10fc:	add    rsi,rax
    10ff:	sar    rsi,1
    1102:	je     1118 <register_tm_clones+0x38>
    1104:	mov    rax,QWORD PTR [rip+0x2ee5]        # 3ff0 <_ITM_registerTMCloneTable>
    110b:	test   rax,rax
    110e:	je     1118 <register_tm_clones+0x38>
    1110:	jmp    rax
    1112:	nop    WORD PTR [rax+rax*1+0x0]
    1118:	ret    
    1119:	nop    DWORD PTR [rax+0x0]

0000000000001120 <__do_global_dtors_aux>:
    1120:	cmp    BYTE PTR [rip+0x2f21],0x0        # 4048 <__TMC_END__>
    1127:	jne    1158 <__do_global_dtors_aux+0x38>
    1129:	push   rbp
    112a:	cmp    QWORD PTR [rip+0x2ec6],0x0        # 3ff8 <__cxa_finalize@GLIBC_2.2.5>
    1132:	mov    rbp,rsp
    1135:	je     1143 <__do_global_dtors_aux+0x23>
    1137:	mov    rdi,QWORD PTR [rip+0x2f02]        # 4040 <__dso_handle>
    113e:	call   1070 <__cxa_finalize@plt>
    1143:	call   10b0 <deregister_tm_clones>
    1148:	mov    BYTE PTR [rip+0x2ef9],0x1        # 4048 <__TMC_END__>
    114f:	pop    rbp
    1150:	ret    
    1151:	nop    DWORD PTR [rax+0x0]
    1158:	ret    
    1159:	nop    DWORD PTR [rax+0x0]

0000000000001160 <frame_dummy>:
    1160:	jmp    10e0 <register_tm_clones>

0000000000001165 <main>:
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>

int main() {
    1165:	push   rbp
    1166:	push   rbx
    1167:	sub    rsp,0x88
    int n;
    char buf[128] = {0};
    116e:	mov    QWORD PTR [rsp],0x0
    1176:	mov    QWORD PTR [rsp+0x8],0x0
    117f:	mov    QWORD PTR [rsp+0x10],0x0
    1188:	mov    QWORD PTR [rsp+0x18],0x0
    1191:	mov    QWORD PTR [rsp+0x20],0x0
    119a:	mov    QWORD PTR [rsp+0x28],0x0
    11a3:	mov    QWORD PTR [rsp+0x30],0x0
    11ac:	mov    QWORD PTR [rsp+0x38],0x0
    11b5:	mov    QWORD PTR [rsp+0x40],0x0
    11be:	mov    QWORD PTR [rsp+0x48],0x0
    11c7:	mov    QWORD PTR [rsp+0x50],0x0
    11d0:	mov    QWORD PTR [rsp+0x58],0x0
    11d9:	mov    QWORD PTR [rsp+0x60],0x0
    11e2:	mov    QWORD PTR [rsp+0x68],0x0
    11eb:	mov    QWORD PTR [rsp+0x70],0x0
    11f4:	mov    QWORD PTR [rsp+0x78],0x0
    int fd = open("/etc/bash.bashrc", O_RDONLY);
    11fd:	mov    esi,0x0
    1202:	lea    rdi,[rip+0xdfb]        # 2004 <_IO_stdin_used+0x4>
    1209:	mov    eax,0x0
    120e:	call   1060 <open@plt>
    1213:	mov    ebx,eax
    n = read(fd, buf, sizeof(buf)/sizeof(buf[0]) - 1);
    1215:	mov    rbp,rsp
    1218:	mov    edx,0x7f
    121d:	mov    rsi,rbp
    1220:	mov    edi,eax
    1222:	call   1050 <read@plt>
    close(fd);
    1227:	mov    edi,ebx
    1229:	call   1040 <close@plt>
    printf("%s\n", buf);
    122e:	mov    rdi,rbp
    1231:	call   1030 <puts@plt>
    return 0;
    1236:	mov    eax,0x0
    123b:	add    rsp,0x88
    1242:	pop    rbx
    1243:	pop    rbp
    1244:	ret    
    1245:	nop    WORD PTR cs:[rax+rax*1+0x0]
    124f:	nop

0000000000001250 <__libc_csu_init>:
    1250:	push   r15
    1252:	lea    r15,[rip+0x2b8f]        # 3de8 <__frame_dummy_init_array_entry>
    1259:	push   r14
    125b:	mov    r14,rdx
    125e:	push   r13
    1260:	mov    r13,rsi
    1263:	push   r12
    1265:	mov    r12d,edi
    1268:	push   rbp
    1269:	lea    rbp,[rip+0x2b80]        # 3df0 <__do_global_dtors_aux_fini_array_entry>
    1270:	push   rbx
    1271:	sub    rbp,r15
    1274:	sub    rsp,0x8
    1278:	call   1000 <_init>
    127d:	sar    rbp,0x3
    1281:	je     129e <__libc_csu_init+0x4e>
    1283:	xor    ebx,ebx
    1285:	nop    DWORD PTR [rax]
    1288:	mov    rdx,r14
    128b:	mov    rsi,r13
    128e:	mov    edi,r12d
    1291:	call   QWORD PTR [r15+rbx*8]
    1295:	add    rbx,0x1
    1299:	cmp    rbp,rbx
    129c:	jne    1288 <__libc_csu_init+0x38>
    129e:	add    rsp,0x8
    12a2:	pop    rbx
    12a3:	pop    rbp
    12a4:	pop    r12
    12a6:	pop    r13
    12a8:	pop    r14
    12aa:	pop    r15
    12ac:	ret    
    12ad:	nop    DWORD PTR [rax]

00000000000012b0 <__libc_csu_fini>:
    12b0:	ret    

Disassembly of section .fini:

00000000000012b4 <_fini>:
    12b4:	sub    rsp,0x8
    12b8:	add    rsp,0x8
    12bc:	ret    
