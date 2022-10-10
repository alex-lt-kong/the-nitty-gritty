# Non-contiguous Memory Accesses

* Four consecutive ints or floats, or two consecutive doubles, may be loaded directly from memory in a single SSE
instruction. But if the four ints are not adjacent, they must be loaded separately using multiple instructions,
which is considerably less efficient.

* The most common examples of non-contiguous memory access are loops with non-unit stride (i.e., stride > 1).

* The compiler rarely vectorizes such loops, unless the amount of computational work is large compared to the overhead
from non-contiguous memory access.

* Note that non-contigupus memory accesses do not always prevent vectorization, but frequently either prevent it or
cause the compiler to decide that vectorization would not be worthwhile.

## Results

* `gcc`:
```
stride: 1,      0.000008106,         152
stride: 4,      0.000016451,           0
```
* `icc`:
```
stride: 1,      0.000007153,         181
stride: 4,      0.000011683,           0
```

## Disassembly analysis

* It is interesting that `gcc` and `icc` handle the 2nd loop (stride == 4 one) differently.

* `icc` does exactly what we expect: it vectorized the 1st loop but didn't vectorize the 2nd.

* `gcc`, however, vectorized both, probably by loading data from memory to SIMD registers separately.

### icc

* 1st loop:
```
  for (int i = 0; i < ARR_SIZE; ++i) {
  4014fc:       48 83 c0 20             add    $0x20,%rax
  401500:       48 3b c6                cmp    %rsi,%rax
  401503:       0f 82 54 ff ff ff       jb     40145d <main+0x1dd>
  401509:       8d 81 01 00 01 00       lea    0x10001(%rcx),%eax
  40150f:       3d 00 00 01 00          cmp    $0x10000,%eax
  401514:       0f 87 93 00 00 00       ja     4015ad <main+0x32d>
  40151a:       f7 da                   neg    %edx
  40151c:       81 c2 00 00 01 00       add    $0x10000,%edx
  401522:       83 fa 08                cmp    $0x8,%edx
  401525:       0f 82 33 02 00 00       jb     40175e <main+0x4de>
  40152b:       33 f6                   xor    %esi,%esi
  40152d:       89 d0                   mov    %edx,%eax
  40152f:       4c 8b 4c 24 10          mov    0x10(%rsp),%r9
  401534:       83 e0 f8                and    $0xfffffff8,%eax
  401537:       66 0f 1f 84 00 00 00    nopw   0x0(%rax,%rax,1)
  40153e:       00 00
    c1[i] += a[i] * b[i];
  401540:       44 8d 84 0e 00 00 01    lea    0x10000(%rsi,%rcx,1),%r8d
  401547:       00
  for (int i = 0; i < ARR_SIZE; ++i) {
  401548:       83 c6 08                add    $0x8,%esi
    c1[i] += a[i] * b[i];
  40154b:       4d 63 c0                movslq %r8d,%r8
  40154e:       f3 43 0f 7e 1c 01       movq   (%r9,%r8,1),%xmm3
  401554:       f3 43 0f 7e 14 38       movq   (%r8,%r15,1),%xmm2
  40155a:       66 0f 60 d9             punpcklbw %xmm1,%xmm3
  40155e:       66 0f 60 d1             punpcklbw %xmm1,%xmm2
  401562:       66 0f d5 da             pmullw %xmm2,%xmm3
  401566:       66 0f db d8             pand   %xmm0,%xmm3
  40156a:       f3 43 0f 7e 24 30       movq   (%r8,%r14,1),%xmm4
  401570:       66 0f 67 d9             packuswb %xmm1,%xmm3
  401574:       66 0f fc e3             paddb  %xmm3,%xmm4
  401578:       66 43 0f d6 24 30       movq   %xmm4,(%r8,%r14,1)
  for (int i = 0; i < ARR_SIZE; ++i) {
  40157e:       3b f0                   cmp    %eax,%esi
  401580:       72 be                   jb     401540 <main+0x2c0>
  401582:       3b c2                   cmp    %edx,%eax
  401584:       73 27                   jae    4015ad <main+0x32d>
  401586:       4c 8b 54 24 10          mov    0x10(%rsp),%r10
    c1[i] += a[i] * b[i];
  40158b:       8d b4 08 00 00 01 00    lea    0x10000(%rax,%rcx,1),%esi
  for (int i = 0; i < ARR_SIZE; ++i) {
  401592:       ff c0                   inc    %eax
    c1[i] += a[i] * b[i];
  401594:       48 63 f6                movslq %esi,%rsi
  401597:       45 0f b6 0c 32          movzbl (%r10,%rsi,1),%r9d
  40159c:       46 0f b6 04 3e          movzbl (%rsi,%r15,1),%r8d
  4015a1:       45 0f af c8             imul   %r8d,%r9d
  4015a5:       46 00 0c 36             add    %r9b,(%rsi,%r14,1)
  for (int i = 0; i < ARR_SIZE; ++i) {
  4015a9:       3b c2                   cmp    %edx,%eax
  4015ab:       72 de                   jb     40158b <main+0x30b>
  }
```
* 2nd loop:
```
  for (int i = 0; i < ARR_SIZE; i += stride) {
  401658:       4c 8b 4c 24 10          mov    0x10(%rsp),%r9
    c2[i] += a[i] * b[i];
  40165d:       41 0f b6 0c c1          movzbl (%r9,%rax,8),%ecx
  401662:       45 0f b6 44 c1 04       movzbl 0x4(%r9,%rax,8),%r8d
  401668:       41 0f b6 14 c7          movzbl (%r15,%rax,8),%edx
  40166d:       41 0f b6 74 c7 04       movzbl 0x4(%r15,%rax,8),%esi
  401673:       0f af ca                imul   %edx,%ecx
  401676:       44 0f af c6             imul   %esi,%r8d
  40167a:       41 00 4c c5 00          add    %cl,0x0(%r13,%rax,8)
  40167f:       45 00 44 c5 04          add    %r8b,0x4(%r13,%rax,8)
  for (int i = 0; i < ARR_SIZE; i += stride) {
  401684:       48 ff c0                inc    %rax
  401687:       48 3d 00 20 00 00       cmp    $0x2000,%rax
  40168d:       72 ce                   jb     40165d <main+0x3dd>
  }
```

### gcc
* 1st loop:
```
  for (int i = 0; i < ARR_SIZE; ++i) {
    1193:       0f 1f 44 00 00          nopl   0x0(%rax,%rax,1)
    c1[i] += a[i] * b[i];
    1198:       f3 41 0f 6f 5c 05 00    movdqu 0x0(%r13,%rax,1),%xmm3
    119f:       f3 41 0f 6f 04 04       movdqu (%r12,%rax,1),%xmm0
    11a5:       66 0f 6f eb             movdqa %xmm3,%xmm5
    11a9:       66 0f 6f d0             movdqa %xmm0,%xmm2
    11ad:       66 0f 60 d0             punpcklbw %xmm0,%xmm2
    11b1:       66 0f 60 eb             punpcklbw %xmm3,%xmm5
    11b5:       66 0f 68 c0             punpckhbw %xmm0,%xmm0
    11b9:       66 0f 68 db             punpckhbw %xmm3,%xmm3
    11bd:       66 0f d5 d5             pmullw %xmm5,%xmm2
    11c1:       66 0f d5 c3             pmullw %xmm3,%xmm0
    11c5:       66 0f db d1             pand   %xmm1,%xmm2
    11c9:       66 0f db c1             pand   %xmm1,%xmm0
    11cd:       66 0f 67 d0             packuswb %xmm0,%xmm2
    11d1:       f3 41 0f 6f 04 06       movdqu (%r14,%rax,1),%xmm0
    11d7:       66 0f fc c2             paddb  %xmm2,%xmm0
    11db:       41 0f 11 04 06          movups %xmm0,(%r14,%rax,1)
  for (int i = 0; i < ARR_SIZE; ++i) {
    11e0:       48 83 c0 10             add    $0x10,%rax
    11e4:       48 3d 00 00 01 00       cmp    $0x10000,%rax
    11ea:       75 ac                   jne    1198 <main+0xe8>
  }
```
* 2nd loop:
```
 const int stride = 4;
  for (int i = 0; i < ARR_SIZE; i += stride) {
    12c8:       0f 1f 84 00 00 00 00    nopl   0x0(%rax,%rax,1)
    12cf:       00
    c2[i] += a[i] * b[i];
    12d0:       f3 0f 6f 19             movdqu (%rcx),%xmm3
    12d4:       48 83 c0 40             add    $0x40,%rax
    12d8:       48 83 c1 40             add    $0x40,%rcx
    12dc:       48 83 c2 40             add    $0x40,%rdx
    12e0:       f3 0f 6f 41 d0          movdqu -0x30(%rcx),%xmm0
    12e5:       f3 0f 6f 51 f0          movdqu -0x10(%rcx),%xmm2
    12ea:       66 0f db d9             pand   %xmm1,%xmm3
    12ee:       f3 0f 6f 6a f0          movdqu -0x10(%rdx),%xmm5
    12f3:       66 0f db c1             pand   %xmm1,%xmm0
    12f7:       66 0f db d1             pand   %xmm1,%xmm2
    12fb:       66 0f 67 d8             packuswb %xmm0,%xmm3
    12ff:       f3 0f 6f 41 e0          movdqu -0x20(%rcx),%xmm0
    1304:       66 0f db e9             pand   %xmm1,%xmm5
    1308:       66 0f db d9             pand   %xmm1,%xmm3
    130c:       66 0f db c1             pand   %xmm1,%xmm0
    1310:       66 0f 67 c2             packuswb %xmm2,%xmm0
    1314:       f3 0f 6f 52 c0          movdqu -0x40(%rdx),%xmm2
    1319:       66 0f db c1             pand   %xmm1,%xmm0
    131d:       66 0f 67 d8             packuswb %xmm0,%xmm3
    1321:       f3 0f 6f 42 d0          movdqu -0x30(%rdx),%xmm0
    1326:       66 0f db d1             pand   %xmm1,%xmm2
    132a:       66 0f db c1             pand   %xmm1,%xmm0
    132e:       66 0f 67 d0             packuswb %xmm0,%xmm2
    1332:       f3 0f 6f 42 e0          movdqu -0x20(%rdx),%xmm0
    1337:       66 0f db d1             pand   %xmm1,%xmm2
    133b:       66 0f db c1             pand   %xmm1,%xmm0
    133f:       66 0f 67 c5             packuswb %xmm5,%xmm0
    1343:       66 0f 6f eb             movdqa %xmm3,%xmm5
    1347:       66 0f db c1             pand   %xmm1,%xmm0
    134b:       66 0f 60 eb             punpcklbw %xmm3,%xmm5
    134f:       66 0f 68 db             punpckhbw %xmm3,%xmm3
    1353:       66 0f 67 d0             packuswb %xmm0,%xmm2
    1357:       66 0f 6f c2             movdqa %xmm2,%xmm0
    135b:       66 0f 60 c2             punpcklbw %xmm2,%xmm0
    135f:       66 0f 68 d2             punpckhbw %xmm2,%xmm2
    1363:       66 0f d5 c5             pmullw %xmm5,%xmm0
    1367:       66 0f d5 d3             pmullw %xmm3,%xmm2
    136b:       f3 0f 6f 58 d0          movdqu -0x30(%rax),%xmm3
    1370:       f3 0f 6f 68 f0          movdqu -0x10(%rax),%xmm5
    1375:       66 0f db d9             pand   %xmm1,%xmm3
    1379:       66 0f db e9             pand   %xmm1,%xmm5
    137d:       66 0f db d1             pand   %xmm1,%xmm2
    1381:       66 0f db c1             pand   %xmm1,%xmm0
    1385:       66 0f 67 c2             packuswb %xmm2,%xmm0
    1389:       f3 0f 6f 50 c0          movdqu -0x40(%rax),%xmm2
    138e:       66 0f db d1             pand   %xmm1,%xmm2
    1392:       66 0f 67 d3             packuswb %xmm3,%xmm2
    1396:       f3 0f 6f 58 e0          movdqu -0x20(%rax),%xmm3
    139b:       66 0f db d1             pand   %xmm1,%xmm2
    139f:       66 0f db d9             pand   %xmm1,%xmm3
    13a3:       66 0f 67 dd             packuswb %xmm5,%xmm3
    13a7:       66 0f db d9             pand   %xmm1,%xmm3
    13ab:       66 0f 67 d3             packuswb %xmm3,%xmm2
    13af:       66 0f fc c2             paddb  %xmm2,%xmm0
    13b3:       66 0f 7e c6             movd   %xmm0,%esi
    13b7:       40 88 70 c0             mov    %sil,-0x40(%rax)
    13bb:       0f 29 84 24 f0 00 00    movaps %xmm0,0xf0(%rsp)
    13c2:       00
    13c3:       0f b6 b4 24 f1 00 00    movzbl 0xf1(%rsp),%esi
    13ca:       00
    13cb:       40 88 70 c4             mov    %sil,-0x3c(%rax)
    13cf:       0f 29 84 24 e0 00 00    movaps %xmm0,0xe0(%rsp)
    13d6:       00
    13d7:       0f b6 b4 24 e2 00 00    movzbl 0xe2(%rsp),%esi
    13de:       00
    13df:       40 88 70 c8             mov    %sil,-0x38(%rax)
    13e3:       0f 29 84 24 d0 00 00    movaps %xmm0,0xd0(%rsp)
    13ea:       00
    13eb:       0f b6 b4 24 d3 00 00    movzbl 0xd3(%rsp),%esi
    13f2:       00
    13f3:       40 88 70 cc             mov    %sil,-0x34(%rax)
    13f7:       0f 29 84 24 c0 00 00    movaps %xmm0,0xc0(%rsp)
    13fe:       00
    13ff:       0f b6 b4 24 c4 00 00    movzbl 0xc4(%rsp),%esi
    1406:       00
    1407:       40 88 70 d0             mov    %sil,-0x30(%rax)
    140b:       0f 29 84 24 b0 00 00    movaps %xmm0,0xb0(%rsp)
    1412:       00
    1413:       0f b6 b4 24 b5 00 00    movzbl 0xb5(%rsp),%esi
    141a:       00
    141b:       40 88 70 d4             mov    %sil,-0x2c(%rax)
    141f:       0f 29 84 24 a0 00 00    movaps %xmm0,0xa0(%rsp)
    1426:       00
    1427:       0f b6 b4 24 a6 00 00    movzbl 0xa6(%rsp),%esi
    142e:       00
    142f:       40 88 70 d8             mov    %sil,-0x28(%rax)
    1433:       0f 29 84 24 90 00 00    movaps %xmm0,0x90(%rsp)
    143a:       00
    143b:       0f b6 b4 24 97 00 00    movzbl 0x97(%rsp),%esi
    1442:       00
    1443:       40 88 70 dc             mov    %sil,-0x24(%rax)
    1447:       0f 29 84 24 80 00 00    movaps %xmm0,0x80(%rsp)
    144e:       00
    144f:       0f b6 b4 24 88 00 00    movzbl 0x88(%rsp),%esi
    1456:       00
    1457:       40 88 70 e0             mov    %sil,-0x20(%rax)
    145b:       0f 29 44 24 70          movaps %xmm0,0x70(%rsp)
    1460:       0f b6 74 24 79          movzbl 0x79(%rsp),%esi
    1465:       40 88 70 e4             mov    %sil,-0x1c(%rax)
    1469:       0f 29 44 24 60          movaps %xmm0,0x60(%rsp)
    146e:       0f b6 74 24 6a          movzbl 0x6a(%rsp),%esi
    1473:       40 88 70 e8             mov    %sil,-0x18(%rax)
    1477:       0f 29 44 24 50          movaps %xmm0,0x50(%rsp)
    147c:       0f b6 74 24 5b          movzbl 0x5b(%rsp),%esi
    1481:       40 88 70 ec             mov    %sil,-0x14(%rax)
    1485:       0f 29 44 24 40          movaps %xmm0,0x40(%rsp)
    148a:       0f b6 74 24 4c          movzbl 0x4c(%rsp),%esi
    148f:       40 88 70 f0             mov    %sil,-0x10(%rax)
    1493:       0f 29 44 24 30          movaps %xmm0,0x30(%rsp)
    1498:       0f b6 74 24 3d          movzbl 0x3d(%rsp),%esi
    149d:       40 88 70 f4             mov    %sil,-0xc(%rax)
    14a1:       0f 29 44 24 20          movaps %xmm0,0x20(%rsp)
    14a6:       0f b6 74 24 2e          movzbl 0x2e(%rsp),%esi
    14ab:       40 88 70 f8             mov    %sil,-0x8(%rax)
    14af:       0f 29 44 24 10          movaps %xmm0,0x10(%rsp)
    14b4:       0f b6 74 24 1f          movzbl 0x1f(%rsp),%esi
    14b9:       40 88 70 fc             mov    %sil,-0x4(%rax)
  for (int i = 0; i < ARR_SIZE; i += stride) {
    14bd:       48 39 c7                cmp    %rax,%rdi
    14c0:       0f 85 0a fe ff ff       jne    12d0 <main+0x220>
  }
```

## Caveat

* Apart from strides 1, 2 and 4, the effect of vectorization/non-vectorization may not be straightforward to observe.
The overall performance can easily be memory-bound, rendering vectorization irrelevant--as CPU is not the bottleneck anyway.