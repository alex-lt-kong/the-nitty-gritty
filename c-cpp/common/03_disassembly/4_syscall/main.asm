Dump of assembler code for function main:
8	int main() {
   0x0000000000001070 <+0>:	push   rbp
   0x000000000000107c <+12>:	sub    rsp,0x70

9	    int n;

10	    char buf[100];

11	    int fd = open ("/etc/passwd", O_RDONLY);
   0x0000000000001071 <+1>:	xor    esi,esi
   0x0000000000001073 <+3>:	lea    rdi,[rip+0xf8a]        # 0x2004
   0x000000000000107a <+10>:	xor    eax,eax
   0x0000000000001080 <+16>:	call   0x1050 <open@plt>
   0x000000000000108d <+29>:	mov    ebp,eax

12	    n = read(fd, buf, 100);
   0x0000000000001085 <+21>:	mov    rsi,rsp
   0x0000000000001088 <+24>:	mov    edx,0x64
   0x000000000000108f <+31>:	mov    edi,eax
   0x0000000000001091 <+33>:	call   0x1040 <read@plt>

13	    close(fd);
   0x0000000000001096 <+38>:	mov    edi,ebp
   0x0000000000001098 <+40>:	call   0x1030 <close@plt>

14	    return 0;

15	}
   0x000000000000109d <+45>:	add    rsp,0x70
   0x00000000000010a1 <+49>:	xor    eax,eax
   0x00000000000010a3 <+51>:	pop    rbp
   0x00000000000010a4 <+52>:	ret    

End of assembler dump.
