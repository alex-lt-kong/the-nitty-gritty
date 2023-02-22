# Hello World

* The very first attempt to show how vectorization can make a difference

* The structure of `main.c` has been carefully designed to avoid memory I/O bottleneck according to:
[this link](https://stackoverflow.com/questions/18159455/why-vectorizing-the-loop-does-not-have-performance-improvement)
and [this link](https://stackoverflow.com/questions/30313600/why-does-my-8m-l3-cache-not-provide-any-benefit-for-arrays-larger-than-1m?noredirect=1&lq=1)

* Results:
```
icc, vectorization on:  3199.74ms
icc, vectorization off: 4400.41ms
gcc, vectorization on:  2610.16ms
gcc, vectorization off: 4861.70ms
```

## Disassembled code

* But how can we be sure if vectorization is really on? Need to dig into assembly code.

### Vectorized versions

#### gcc

`gdb -batch -ex 'file ./main-gcc-on.out' -ex 'disassemble /m linear_func' | cut -d " " -f 5-`
```
...
results[i] = a * arr[i] + b;
<+88>:  movdqu (%rdx,%rax,1),%xmm0
<+93>:  movdqu (%rdx,%rax,1),%xmm1
<+98>:  psrlq  $0x20,%xmm0
<+103>: pmuludq %xmm2,%xmm1
<+107>: pmuludq %xmm3,%xmm0
<+111>: pshufd $0x8,%xmm1,%xmm1
<+116>: pshufd $0x8,%xmm0,%xmm0
<+121>: punpckldq %xmm0,%xmm1
<+125>: movdqa %xmm1,%xmm0
<+129>: paddd  %xmm4,%xmm0
<+133>: movups %xmm0,(%rcx,%rax,1)
<+159>: mov    (%rdx,%rax,4),%r9d
<+163>: imul   %edi,%r9d
<+167>: add    %esi,%r9d
<+170>: mov    %r9d,(%rcx,%rax,4)
<+186>: mov    (%rdx,%r9,4),%r10d
<+195>: imul   %edi,%r10d
<+199>: add    %esi,%r10d
<+202>: mov    %r10d,(%rcx,%r9,4)
<+211>: imul   (%rdx,%rax,4),%edi
<+215>: add    %edi,%esi
<+217>: mov    %esi,(%rcx,%rax,4)
<+240>: mov    (%rdx,%rax,1),%r9d
<+244>: imul   %edi,%r9d
<+248>: add    %esi,%r9d
<+251>: mov    %r9d,(%rcx,%rax,1)
...
```

#### icc

`gdb -batch -ex 'file ./main-icc-on.out' -ex 'disassemble /m linear_func' | cut -d " " -f 5-`
```
...
results[i] = a * arr[i] + b;
<+19>:  mov    %rcx,%rax
<+30>:  sub    %rdx,%rax
<+38>:  neg    %rax
<+128>: mov    (%rdx,%r10,4),%r12d
<+135>: imul   %edi,%r12d
<+139>: add    %esi,%r12d
<+142>: mov    %r12d,(%rcx,%r10,4)
<+159>: mov    %r11d,%r9d
<+162>: lea    (%rdx,%r11,4),%r10
<+188>: movdqa %xmm3,%xmm2
<+197>: psrlq  $0x20,%xmm2
<+202>: movdqu 0x393e(%rip),%xmm0        # 0x405010
<+210>: movdqu (%rdx,%r11,4),%xmm4
<+216>: movdqa %xmm3,%xmm5
<+224>: pmuludq %xmm4,%xmm5
<+228>: psrlq  $0x20,%xmm4
<+233>: pmuludq %xmm2,%xmm4
<+237>: pand   %xmm0,%xmm5
<+241>: psllq  $0x20,%xmm4
<+246>: por    %xmm4,%xmm5
<+250>: paddd  %xmm1,%xmm5
<+254>: movdqu %xmm5,(%rcx,%r11,4)
<+284>: movdqa %xmm3,%xmm2
<+293>: psrlq  $0x20,%xmm2
<+298>: movdqu 0x38de(%rip),%xmm0        # 0x405010
<+306>: movdqu (%rdx,%r11,4),%xmm4
<+312>: movdqa %xmm3,%xmm5
<+316>: pmuludq %xmm4,%xmm5
<+320>: psrlq  $0x20,%xmm4
<+325>: pmuludq %xmm2,%xmm4
<+329>: pand   %xmm0,%xmm5
<+333>: psllq  $0x20,%xmm4
<+338>: por    %xmm4,%xmm5
<+346>: paddd  %xmm1,%xmm5
<+350>: movdqu %xmm5,(%rcx,%r11,4)
<+378>: mov    (%rdx,%r10,4),%eax
<+385>: imul   %edi,%eax
<+388>: add    %esi,%eax
<+390>: mov    %eax,(%rcx,%r10,4)
<+420>: movslq %r9d,%r9
<+423>: mov    (%rdx,%r9,8),%eax
<+427>: imul   %edi,%eax
<+430>: add    %esi,%eax
<+432>: mov    %eax,(%rcx,%r9,8)
<+436>: mov    0x4(%rdx,%r9,8),%r11d
<+441>: imul   %edi,%r11d
<+445>: add    %esi,%r11d
<+448>: mov    %r11d,0x4(%rcx,%r9,8)
<+481>: mov    (%rdx,%rax,4),%edx
<+484>: imul   %edx,%edi
<+487>: add    %edi,%esi
<+489>: mov    %esi,(%rcx,%rax,4)
...
```
* What are `MMX` instructions? https://en.wikipedia.org/wiki/MMX_(instruction_set)
* How about mnemonics such as `pand`? https://docs.oracle.com/cd/E18752_01/html/817-5477/eojdc.html
* One can notice that both `icc` and `gcc` use SIMD instructions extensively to vectorize the loop. However, `icc`
outperforms `gcc` by a large margin--`gcc`'s vectorization barely makes any difference (compared with its own
non-vectorized version).

### Non-vectorized versions

#### gcc

`gdb -batch -ex 'file ./main-gcc-no.out' -ex 'disassemble /m linear_func' | cut -d " " -f 5-`

```
...
results[i] = a * arr[i] + b;
<+16>:  mov    (%r9,%rax,1),%edx
<+20>:  imul   %edi,%edx
<+23>:  add    %esi,%edx
<+25>:  mov    %edx,(%rcx,%rax,1)
...
 ```

#### icc

`gdb -batch -ex 'file ./main-icc-no.out' -ex 'disassemble /m linear_func' | cut -d " " -f 5-`

```
...
results[i] = a * arr[i] + b;
<+19>:  mov    %rcx,%rax
<+30>:  sub    %rdx,%rax
<+38>:  neg    %rax
<+71>:  movslq %r10d,%rax
<+77>:  shl    $0x5,%rax
<+81>:  mov    (%rdx,%rax,1),%r11d
<+85>:  imul   %edi,%r11d
<+89>:  add    %esi,%r11d
<+92>:  mov    %r11d,(%rcx,%rax,1)
<+96>:  mov    0x4(%rdx,%rax,1),%r11d
<+101>: imul   %edi,%r11d
<+105>: add    %esi,%r11d
<+108>: mov    %r11d,0x4(%rcx,%rax,1)
<+113>: mov    0x8(%rdx,%rax,1),%r11d
<+118>: imul   %edi,%r11d
<+122>: add    %esi,%r11d
<+125>: mov    %r11d,0x8(%rcx,%rax,1)
<+130>: mov    0xc(%rdx,%rax,1),%r11d
<+135>: imul   %edi,%r11d
<+139>: add    %esi,%r11d
<+142>: mov    %r11d,0xc(%rcx,%rax,1)
<+147>: mov    0x10(%rdx,%rax,1),%r11d
<+152>: imul   %edi,%r11d
<+156>: add    %esi,%r11d
<+159>: mov    %r11d,0x10(%rcx,%rax,1)
<+164>: mov    0x14(%rdx,%rax,1),%r11d
<+169>: imul   %edi,%r11d
<+173>: add    %esi,%r11d
<+176>: mov    %r11d,0x14(%rcx,%rax,1)
<+181>: mov    0x18(%rdx,%rax,1),%r11d
<+186>: imul   %edi,%r11d
<+190>: add    %esi,%r11d
<+193>: mov    %r11d,0x18(%rcx,%rax,1)
<+198>: mov    0x1c(%rdx,%rax,1),%r11d
<+203>: imul   %edi,%r11d
<+207>: add    %esi,%r11d
<+210>: mov    %r11d,0x1c(%rcx,%rax,1)
<+260>: mov    0x14(%rdx,%rax,4),%r8d
<+265>: imul   %edi,%r8d
<+269>: add    %esi,%r8d
<+272>: mov    %r8d,0x14(%rcx,%rax,4)
<+277>: mov    0x10(%rdx,%rax,4),%r8d
<+282>: imul   %edi,%r8d
<+286>: add    %esi,%r8d
<+289>: mov    %r8d,0x10(%rcx,%rax,4)
<+294>: mov    0xc(%rdx,%rax,4),%r8d
<+299>: imul   %edi,%r8d
<+303>: add    %esi,%r8d
<+306>: mov    %r8d,0xc(%rcx,%rax,4)
<+311>: mov    0x8(%rdx,%rax,4),%r8d
<+316>: imul   %edi,%r8d
<+320>: add    %esi,%r8d
<+323>: mov    %r8d,0x8(%rcx,%rax,4)
<+328>: mov    0x4(%rdx,%rax,4),%r8d
<+333>: imul   %edi,%r8d
<+337>: add    %esi,%r8d
<+340>: mov    %r8d,0x4(%rcx,%rax,4)
<+345>: mov    (%rdx,%rax,4),%r8d
<+349>: imul   %edi,%r8d
<+353>: add    %esi,%r8d
<+356>: mov    %r8d,(%rcx,%rax,4)
<+360>: mov    -0x4(%rdx,%rax,4),%edx
<+364>: imul   %edx,%edi
<+367>: add    %edi,%esi
<+369>: mov    %esi,-0x4(%rcx,%rax,4)
<+373>: jmp    0x40163f <linear_func+463>
<+391>: movslq %r9d,%r9
<+394>: mov    (%rdx,%r9,8),%eax
<+398>: imul   %edi,%eax
<+401>: add    %esi,%eax
<+403>: mov    %eax,(%rcx,%r9,8)
<+407>: mov    0x4(%rdx,%r9,8),%r11d
<+412>: imul   %edi,%r11d
<+416>: add    %esi,%r11d
<+419>: mov    %r11d,0x4(%rcx,%r9,8)
<+452>: mov    (%rdx,%rax,4),%edx
<+455>: imul   %edx,%edi
<+458>: add    %edi,%esi
<+460>: mov    %esi,(%rcx,%rax,4)
...
```