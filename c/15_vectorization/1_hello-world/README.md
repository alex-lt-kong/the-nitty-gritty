# Hello World

* The very first attempt to show how vectorization can make a difference

* Results:
```
icc, vectorization on:  Average: 66ms, std: 16.943430
icc, vectorization off: Average: 111ms, std: 19.272556
gcc, vectorization on:  Average: 102ms, std: 13.690416
gcc, vectorization off: Average: 114ms, std: 20.553469
```

## Disassembled code

* But how can we be sure if vectorization is really on? Need to dig into assembly code.

### Vectorized versions

#### icc
```
gdb ./main-icc-on.out
set disassembly-flavor intel
layout split
break main.c:11
run
ni
ni
ni...
```
```
 0x4014f6 <main+550>     movdqu xmm4,XMMWORD PTR [r9+rsi*4]
 0x4014fc <main+556>     movdqa xmm5,xmm3
 0x401500 <main+560>     pmuludq xmm5,xmm4
 0x401504 <main+564>     psrlq  xmm4,0x20
 0x401509 <main+569>     pmuludq xmm4,xmm2
 0x40150d <main+573>     pand   xmm5,xmm0
 0x401511 <main+577>     psllq  xmm4,0x20 
 0x401516 <main+582>     por    xmm5,xmm4 
 0x40151a <main+586>     add    eax,0x4
>0x40151d <main+589>     paddd  xmm5,xmm1 
 0x401521 <main+593>     movntdq XMMWORD PTR [r8+rsi*4],xmm5
 0x401527 <main+599>     add    rsi,0x4
 0x40152b <main+603>     cmp    rax,rdx
 0x40152e <main+606>     jb     0x4014f6 <main+550> 
```

#### gcc

```
gdb ./main-gcc-on.out
set disassembly-flavor intel
layout split
break main.c:11
run
ni
ni
ni...
```
```
 0x5555555551e0 <main+240>       movdqu xmm0,XMMWORD PTR [r13+rax*1+0x0]
 0x5555555551e7 <main+247>       movdqu xmm1,XMMWORD PTR [r13+rax*1+0x0]
 0x5555555551ee <main+254>       psrlq  xmm0,0x20
 0x5555555551f3 <main+259>       pmuludq xmm1,xmm3
 0x5555555551f7 <main+263>       pmuludq xmm0,xmm4
 0x5555555551fb <main+267>       pshufd xmm1,xmm1,0x8
 0x555555555200 <main+272>       pshufd xmm0,xmm0,0x8
 0x555555555205 <main+277>       punpckldq xmm1,xmm0 
 0x555555555209 <main+281>       paddd  xmm1,xmm2
 0x55555555520d <main+285>       movups XMMWORD PTR [rbp+rax*1+0x0],xmm1
>0x555555555212 <main+290>       add    rax,0x10
 0x555555555216 <main+294>       cmp    rax,0x20000000
 0x55555555521c <main+300>       jne    0x5555555551e0 <main+240>

```
* What are `MMX` instructions? https://en.wikipedia.org/wiki/MMX_(instruction_set)
* How about mnemonics such as `pand`? https://docs.oracle.com/cd/E18752_01/html/817-5477/eojdc.html
* One can notice that both `icc` and `gcc` use SIMD instructions extensively to vectorize the loop. However, `icc`
outperforms `gcc` by a large margin--`gcc`'s vectorization barely makes any difference (compared with its own
non-vectorized version).

### Non-vectorized versions

```
gdb ./main-icc-no.out
set disassembly-flavor intel
layout split
break main.c:11
run
ni
ni
ni...
```
```
 0x4013d0 <main+256>     mov    r8d,DWORD PTR [r10+rcx*8]
 0x4013d4 <main+260>     inc    esi
>0x4013d6 <main+262>     mov    r9d,DWORD PTR [r10+rcx*8+0x4]
 0x4013db <main+267>     imul   r8d,r14d
 0x4013df <main+271>     imul   r9d
 0x4013e3 <main+275>     add    r8d,eax
 0x4013e6 <main+278>     add    r9d,eax
 0x4013e9 <main+281>     mov    DWORD PTR [rbx+rcx*8],r8d
 0x4013ed <main+285>     mov    DWORD PTR [rbx+rcx*8+0x4],r9d
 0x4013f2 <main+290>     inc    rcx
 0x4013f5 <main+293>     cmp    esi,0x4000000
 ```