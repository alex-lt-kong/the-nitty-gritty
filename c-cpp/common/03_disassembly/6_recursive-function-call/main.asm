
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

0000000000001050 <main>:
#include <stdio.h>

#include "factorial.h"

int main() {
    1050:	sub    rsp,0x8
    printf("%lu\n", factorial(7));
    1054:	mov    edi,0x7
    1059:	call   1165 <factorial>
    105e:	lea    rdi,[rip+0xf9f]        # 2004 <_IO_stdin_used+0x4>
    1065:	mov    rsi,rax
    1068:	xor    eax,eax
    106a:	call   1030 <printf@plt>
    return 0;
    106f:	xor    eax,eax
    1071:	add    rsp,0x8
    1075:	ret    
    1076:	nop    WORD PTR cs:[rax+rax*1+0x0]

0000000000001080 <_start>:
    1080:	xor    ebp,ebp
    1082:	mov    r9,rdx
    1085:	pop    rsi
    1086:	mov    rdx,rsp
    1089:	and    rsp,0xfffffffffffffff0
    108d:	push   rax
    108e:	push   rsp
    108f:	lea    r8,[rip+0x16a]        # 1200 <__libc_csu_fini>
    1096:	lea    rcx,[rip+0x103]        # 11a0 <__libc_csu_init>
    109d:	lea    rdi,[rip+0xffffffffffffffac]        # 1050 <main>
    10a4:	call   QWORD PTR [rip+0x2f36]        # 3fe0 <__libc_start_main@GLIBC_2.2.5>
    10aa:	hlt    
    10ab:	nop    DWORD PTR [rax+rax*1+0x0]

00000000000010b0 <deregister_tm_clones>:
    10b0:	lea    rdi,[rip+0x2f79]        # 4030 <__TMC_END__>
    10b7:	lea    rax,[rip+0x2f72]        # 4030 <__TMC_END__>
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
    10e0:	lea    rdi,[rip+0x2f49]        # 4030 <__TMC_END__>
    10e7:	lea    rsi,[rip+0x2f42]        # 4030 <__TMC_END__>
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
    1120:	cmp    BYTE PTR [rip+0x2f09],0x0        # 4030 <__TMC_END__>
    1127:	jne    1158 <__do_global_dtors_aux+0x38>
    1129:	push   rbp
    112a:	cmp    QWORD PTR [rip+0x2ec6],0x0        # 3ff8 <__cxa_finalize@GLIBC_2.2.5>
    1132:	mov    rbp,rsp
    1135:	je     1143 <__do_global_dtors_aux+0x23>
    1137:	mov    rdi,QWORD PTR [rip+0x2eea]        # 4028 <__dso_handle>
    113e:	call   1040 <__cxa_finalize@plt>
    1143:	call   10b0 <deregister_tm_clones>
    1148:	mov    BYTE PTR [rip+0x2ee1],0x1        # 4030 <__TMC_END__>
    114f:	pop    rbp
    1150:	ret    
    1151:	nop    DWORD PTR [rax+0x0]
    1158:	ret    
    1159:	nop    DWORD PTR [rax+0x0]

0000000000001160 <frame_dummy>:
    1160:	jmp    10e0 <register_tm_clones>

0000000000001165 <factorial>:
    1165:	push   rbp
    1166:	mov    rbp,rsp
    1169:	sub    rsp,0x10
    116d:	mov    QWORD PTR [rbp-0x8],rdi
    1171:	cmp    QWORD PTR [rbp-0x8],0x1
    1176:	jne    117e <factorial+0x19>
    1178:	mov    rax,QWORD PTR [rbp-0x8]
    117c:	jmp    1193 <factorial+0x2e>
    117e:	mov    rax,QWORD PTR [rbp-0x8]
    1182:	sub    rax,0x1
    1186:	mov    rdi,rax
    1189:	call   1165 <factorial>
    118e:	imul   rax,QWORD PTR [rbp-0x8]
    1193:	leave  
    1194:	ret    
    1195:	nop    WORD PTR cs:[rax+rax*1+0x0]
    119f:	nop

00000000000011a0 <__libc_csu_init>:
    11a0:	push   r15
    11a2:	lea    r15,[rip+0x2c3f]        # 3de8 <__frame_dummy_init_array_entry>
    11a9:	push   r14
    11ab:	mov    r14,rdx
    11ae:	push   r13
    11b0:	mov    r13,rsi
    11b3:	push   r12
    11b5:	mov    r12d,edi
    11b8:	push   rbp
    11b9:	lea    rbp,[rip+0x2c30]        # 3df0 <__do_global_dtors_aux_fini_array_entry>
    11c0:	push   rbx
    11c1:	sub    rbp,r15
    11c4:	sub    rsp,0x8
    11c8:	call   1000 <_init>
    11cd:	sar    rbp,0x3
    11d1:	je     11ee <__libc_csu_init+0x4e>
    11d3:	xor    ebx,ebx
    11d5:	nop    DWORD PTR [rax]
    11d8:	mov    rdx,r14
    11db:	mov    rsi,r13
    11de:	mov    edi,r12d
    11e1:	call   QWORD PTR [r15+rbx*8]
    11e5:	add    rbx,0x1
    11e9:	cmp    rbp,rbx
    11ec:	jne    11d8 <__libc_csu_init+0x38>
    11ee:	add    rsp,0x8
    11f2:	pop    rbx
    11f3:	pop    rbp
    11f4:	pop    r12
    11f6:	pop    r13
    11f8:	pop    r14
    11fa:	pop    r15
    11fc:	ret    
    11fd:	nop    DWORD PTR [rax]

0000000000001200 <__libc_csu_fini>:
    1200:	ret    

Disassembly of section .fini:

0000000000001204 <_fini>:
    1204:	sub    rsp,0x8
    1208:	add    rsp,0x8
    120c:	ret    
