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

* `objdump --disassembler-options "intel" --disassemble="main" -S <executable>` is used to generate assembly code.

* It is interesting that `gcc` and `icc` handle the 2nd loop (stride == 4 one) differently.

* `icc` does exactly what we expect: it vectorized the 1st loop but didn't vectorize the 2nd.

* `gcc`, however, vectorized both, probably by loading data from memory to SIMD registers separately.

### icc

* 1st loop:
```Assembly
  for (int i = 0; i < ARR_SIZE; ++i) {
  401548:       83 c6 08                add    esi,0x8
    c1[i] += a[i] * b[i];
  40154b:       4d 63 c0                movsxd r8,r8d
  40154e:       f3 43 0f 7e 1c 01       movq   xmm3,QWORD PTR [r9+r8*1]
  401554:       f3 43 0f 7e 14 38       movq   xmm2,QWORD PTR [r8+r15*1]
  40155a:       66 0f 60 d9             punpcklbw xmm3,xmm1
  40155e:       66 0f 60 d1             punpcklbw xmm2,xmm1
  401562:       66 0f d5 da             pmullw xmm3,xmm2
  401566:       66 0f db d8             pand   xmm3,xmm0
  40156a:       f3 43 0f 7e 24 30       movq   xmm4,QWORD PTR [r8+r14*1]
  401570:       66 0f 67 d9             packuswb xmm3,xmm1
  401574:       66 0f fc e3             paddb  xmm4,xmm3
  401578:       66 43 0f d6 24 30       movq   QWORD PTR [r8+r14*1],xmm4
  for (int i = 0; i < ARR_SIZE; ++i) {
  40157e:       3b f0                   cmp    esi,eax
  401580:       72 be                   jb     401540 <main+0x2c0>
  401582:       3b c2                   cmp    eax,edx
  401584:       73 27                   jae    4015ad <main+0x32d>
  401586:       4c 8b 54 24 10          mov    r10,QWORD PTR [rsp+0x10]
    c1[i] += a[i] * b[i];
  40158b:       8d b4 08 00 00 01 00    lea    esi,[rax+rcx*1+0x10000]
  for (int i = 0; i < ARR_SIZE; ++i) {
  401592:       ff c0                   inc    eax
    c1[i] += a[i] * b[i];
  401594:       48 63 f6                movsxd rsi,esi
  401597:       45 0f b6 0c 32          movzx  r9d,BYTE PTR [r10+rsi*1]
  40159c:       46 0f b6 04 3e          movzx  r8d,BYTE PTR [rsi+r15*1]
  4015a1:       45 0f af c8             imul   r9d,r8d
  4015a5:       46 00 0c 36             add    BYTE PTR [rsi+r14*1],r9b
  for (int i = 0; i < ARR_SIZE; ++i) {
  4015a9:       3b c2                   cmp    eax,edx
  4015ab:       72 de                   jb     40158b <main+0x30b>
  }
```
* 2nd loop:
```
const int stride = 4;
  for (int i = 0; i < ARR_SIZE; i += stride) {
  401651:       33 c0                   xor    eax,eax
  t0 = ts.tv_sec + ts.tv_nsec / 1000.0 / 1000.0 / 1000.0;
  401653:       48 8b 5c 24 08          mov    rbx,QWORD PTR [rsp+0x8]
  for (int i = 0; i < ARR_SIZE; i += stride) {
  401658:       4c 8b 4c 24 10          mov    r9,QWORD PTR [rsp+0x10]
    c2[i] += a[i] * b[i];
  40165d:       41 0f b6 0c c1          movzx  ecx,BYTE PTR [r9+rax*8]
  401662:       45 0f b6 44 c1 04       movzx  r8d,BYTE PTR [r9+rax*8+0x4]
  401668:       41 0f b6 14 c7          movzx  edx,BYTE PTR [r15+rax*8]
  40166d:       41 0f b6 74 c7 04       movzx  esi,BYTE PTR [r15+rax*8+0x4]
  401673:       0f af ca                imul   ecx,edx
  401676:       44 0f af c6             imul   r8d,esi
  40167a:       41 00 4c c5 00          add    BYTE PTR [r13+rax*8+0x0],cl
  40167f:       45 00 44 c5 04          add    BYTE PTR [r13+rax*8+0x4],r8b
  for (int i = 0; i < ARR_SIZE; i += stride) {
  401684:       48 ff c0                inc    rax
  401687:       48 3d 00 20 00 00       cmp    rax,0x2000
  40168d:       72 ce                   jb     40165d <main+0x3dd>
  }
```

### gcc
* 1st loop:
```
  for (int i = 0; i < ARR_SIZE; ++i) {
    1193:       0f 1f 44 00 00          nop    DWORD PTR [rax+rax*1+0x0]
    c1[i] += a[i] * b[i];
    1198:       f3 41 0f 6f 5c 05 00    movdqu xmm3,XMMWORD PTR [r13+rax*1+0x0]
    119f:       f3 41 0f 6f 04 04       movdqu xmm0,XMMWORD PTR [r12+rax*1]
    11a5:       66 0f 6f eb             movdqa xmm5,xmm3
    11a9:       66 0f 6f d0             movdqa xmm2,xmm0
    11ad:       66 0f 60 d0             punpcklbw xmm2,xmm0
    11b1:       66 0f 60 eb             punpcklbw xmm5,xmm3
    11b5:       66 0f 68 c0             punpckhbw xmm0,xmm0
    11b9:       66 0f 68 db             punpckhbw xmm3,xmm3
    11bd:       66 0f d5 d5             pmullw xmm2,xmm5
    11c1:       66 0f d5 c3             pmullw xmm0,xmm3
    11c5:       66 0f db d1             pand   xmm2,xmm1
    11c9:       66 0f db c1             pand   xmm0,xmm1
    11cd:       66 0f 67 d0             packuswb xmm2,xmm0
    11d1:       f3 41 0f 6f 04 06       movdqu xmm0,XMMWORD PTR [r14+rax*1]
    11d7:       66 0f fc c2             paddb  xmm0,xmm2
    11db:       41 0f 11 04 06          movups XMMWORD PTR [r14+rax*1],xmm0
  for (int i = 0; i < ARR_SIZE; ++i) {
    11e0:       48 83 c0 10             add    rax,0x10
    11e4:       48 3d 00 00 01 00       cmp    rax,0x10000
    11ea:       75 ac                   jne    1198 <main+0xe8>
  }
```
* 2nd loop:
```
  const int stride = 4;
  for (int i = 0; i < ARR_SIZE; i += stride) {
    12c8:       0f 1f 84 00 00 00 00    nop    DWORD PTR [rax+rax*1+0x0]
    12cf:       00
    c2[i] += a[i] * b[i];
    12d0:       f3 0f 6f 19             movdqu xmm3,XMMWORD PTR [rcx]
    12d4:       48 83 c0 40             add    rax,0x40
    12d8:       48 83 c1 40             add    rcx,0x40
    12dc:       48 83 c2 40             add    rdx,0x40
    12e0:       f3 0f 6f 41 d0          movdqu xmm0,XMMWORD PTR [rcx-0x30]
    12e5:       f3 0f 6f 51 f0          movdqu xmm2,XMMWORD PTR [rcx-0x10]
    12ea:       66 0f db d9             pand   xmm3,xmm1
    12ee:       f3 0f 6f 6a f0          movdqu xmm5,XMMWORD PTR [rdx-0x10]
    12f3:       66 0f db c1             pand   xmm0,xmm1
    12f7:       66 0f db d1             pand   xmm2,xmm1
    12fb:       66 0f 67 d8             packuswb xmm3,xmm0
    12ff:       f3 0f 6f 41 e0          movdqu xmm0,XMMWORD PTR [rcx-0x20]
    1304:       66 0f db e9             pand   xmm5,xmm1
    1308:       66 0f db d9             pand   xmm3,xmm1
    130c:       66 0f db c1             pand   xmm0,xmm1
    1310:       66 0f 67 c2             packuswb xmm0,xmm2
    1314:       f3 0f 6f 52 c0          movdqu xmm2,XMMWORD PTR [rdx-0x40]
    1319:       66 0f db c1             pand   xmm0,xmm1
    131d:       66 0f 67 d8             packuswb xmm3,xmm0
    1321:       f3 0f 6f 42 d0          movdqu xmm0,XMMWORD PTR [rdx-0x30]
    1326:       66 0f db d1             pand   xmm2,xmm1
    132a:       66 0f db c1             pand   xmm0,xmm1
    132e:       66 0f 67 d0             packuswb xmm2,xmm0
    1332:       f3 0f 6f 42 e0          movdqu xmm0,XMMWORD PTR [rdx-0x20]
    1337:       66 0f db d1             pand   xmm2,xmm1
    133b:       66 0f db c1             pand   xmm0,xmm1
    133f:       66 0f 67 c5             packuswb xmm0,xmm5
    1343:       66 0f 6f eb             movdqa xmm5,xmm3
    1347:       66 0f db c1             pand   xmm0,xmm1
    134b:       66 0f 60 eb             punpcklbw xmm5,xmm3
    134f:       66 0f 68 db             punpckhbw xmm3,xmm3
    1353:       66 0f 67 d0             packuswb xmm2,xmm0
    1357:       66 0f 6f c2             movdqa xmm0,xmm2
    135b:       66 0f 60 c2             punpcklbw xmm0,xmm2
    135f:       66 0f 68 d2             punpckhbw xmm2,xmm2
    1363:       66 0f d5 c5             pmullw xmm0,xmm5
    1367:       66 0f d5 d3             pmullw xmm2,xmm3
    136b:       f3 0f 6f 58 d0          movdqu xmm3,XMMWORD PTR [rax-0x30]
    1370:       f3 0f 6f 68 f0          movdqu xmm5,XMMWORD PTR [rax-0x10]
    1375:       66 0f db d9             pand   xmm3,xmm1
    1379:       66 0f db e9             pand   xmm5,xmm1
    137d:       66 0f db d1             pand   xmm2,xmm1
    1381:       66 0f db c1             pand   xmm0,xmm1
    1385:       66 0f 67 c2             packuswb xmm0,xmm2
    1389:       f3 0f 6f 50 c0          movdqu xmm2,XMMWORD PTR [rax-0x40]
    138e:       66 0f db d1             pand   xmm2,xmm1
    1392:       66 0f 67 d3             packuswb xmm2,xmm3
    1396:       f3 0f 6f 58 e0          movdqu xmm3,XMMWORD PTR [rax-0x20]
    139b:       66 0f db d1             pand   xmm2,xmm1
    139f:       66 0f db d9             pand   xmm3,xmm1
    13a3:       66 0f 67 dd             packuswb xmm3,xmm5
    13a7:       66 0f db d9             pand   xmm3,xmm1
    13ab:       66 0f 67 d3             packuswb xmm2,xmm3
    13af:       66 0f fc c2             paddb  xmm0,xmm2
    13b3:       66 0f 7e c6             movd   esi,xmm0
    13b7:       40 88 70 c0             mov    BYTE PTR [rax-0x40],sil
    13bb:       0f 29 84 24 f0 00 00    movaps XMMWORD PTR [rsp+0xf0],xmm0
    13c2:       00
    13c3:       0f b6 b4 24 f1 00 00    movzx  esi,BYTE PTR [rsp+0xf1]
    13ca:       00
    13cb:       40 88 70 c4             mov    BYTE PTR [rax-0x3c],sil
    13cf:       0f 29 84 24 e0 00 00    movaps XMMWORD PTR [rsp+0xe0],xmm0
    13d6:       00
    13d7:       0f b6 b4 24 e2 00 00    movzx  esi,BYTE PTR [rsp+0xe2]
    13de:       00
    13df:       40 88 70 c8             mov    BYTE PTR [rax-0x38],sil
    13e3:       0f 29 84 24 d0 00 00    movaps XMMWORD PTR [rsp+0xd0],xmm0
    13ea:       00
    13eb:       0f b6 b4 24 d3 00 00    movzx  esi,BYTE PTR [rsp+0xd3]
    13f2:       00
    13f3:       40 88 70 cc             mov    BYTE PTR [rax-0x34],sil
    13f7:       0f 29 84 24 c0 00 00    movaps XMMWORD PTR [rsp+0xc0],xmm0
    13fe:       00
    13ff:       0f b6 b4 24 c4 00 00    movzx  esi,BYTE PTR [rsp+0xc4]
    1406:       00
    1407:       40 88 70 d0             mov    BYTE PTR [rax-0x30],sil
    140b:       0f 29 84 24 b0 00 00    movaps XMMWORD PTR [rsp+0xb0],xmm0
    1412:       00
    1413:       0f b6 b4 24 b5 00 00    movzx  esi,BYTE PTR [rsp+0xb5]
    141a:       00
    141b:       40 88 70 d4             mov    BYTE PTR [rax-0x2c],sil
    141f:       0f 29 84 24 a0 00 00    movaps XMMWORD PTR [rsp+0xa0],xmm0
    1426:       00
    1427:       0f b6 b4 24 a6 00 00    movzx  esi,BYTE PTR [rsp+0xa6]
    142e:       00
    142f:       40 88 70 d8             mov    BYTE PTR [rax-0x28],sil
    1433:       0f 29 84 24 90 00 00    movaps XMMWORD PTR [rsp+0x90],xmm0
    143a:       00
    143b:       0f b6 b4 24 97 00 00    movzx  esi,BYTE PTR [rsp+0x97]
    1442:       00
    1443:       40 88 70 dc             mov    BYTE PTR [rax-0x24],sil
    1447:       0f 29 84 24 80 00 00    movaps XMMWORD PTR [rsp+0x80],xmm0
    144e:       00
    144f:       0f b6 b4 24 88 00 00    movzx  esi,BYTE PTR [rsp+0x88]
    1456:       00
    1457:       40 88 70 e0             mov    BYTE PTR [rax-0x20],sil
    145b:       0f 29 44 24 70          movaps XMMWORD PTR [rsp+0x70],xmm0
    1460:       0f b6 74 24 79          movzx  esi,BYTE PTR [rsp+0x79]
    1465:       40 88 70 e4             mov    BYTE PTR [rax-0x1c],sil
    1469:       0f 29 44 24 60          movaps XMMWORD PTR [rsp+0x60],xmm0
    146e:       0f b6 74 24 6a          movzx  esi,BYTE PTR [rsp+0x6a]
    1473:       40 88 70 e8             mov    BYTE PTR [rax-0x18],sil
    1477:       0f 29 44 24 50          movaps XMMWORD PTR [rsp+0x50],xmm0
    147c:       0f b6 74 24 5b          movzx  esi,BYTE PTR [rsp+0x5b]
    1481:       40 88 70 ec             mov    BYTE PTR [rax-0x14],sil
    1485:       0f 29 44 24 40          movaps XMMWORD PTR [rsp+0x40],xmm0
    148a:       0f b6 74 24 4c          movzx  esi,BYTE PTR [rsp+0x4c]
    148f:       40 88 70 f0             mov    BYTE PTR [rax-0x10],sil
    1493:       0f 29 44 24 30          movaps XMMWORD PTR [rsp+0x30],xmm0
    1498:       0f b6 74 24 3d          movzx  esi,BYTE PTR [rsp+0x3d]
    149d:       40 88 70 f4             mov    BYTE PTR [rax-0xc],sil
    14a1:       0f 29 44 24 20          movaps XMMWORD PTR [rsp+0x20],xmm0
    14a6:       0f b6 74 24 2e          movzx  esi,BYTE PTR [rsp+0x2e]
    14ab:       40 88 70 f8             mov    BYTE PTR [rax-0x8],sil
    14af:       0f 29 44 24 10          movaps XMMWORD PTR [rsp+0x10],xmm0
    14b4:       0f b6 74 24 1f          movzx  esi,BYTE PTR [rsp+0x1f]
    14b9:       40 88 70 fc             mov    BYTE PTR [rax-0x4],sil
  for (int i = 0; i < ARR_SIZE; i += stride) {
    14bd:       48 39 c7                cmp    rdi,rax
    14c0:       0f 85 0a fe ff ff       jne    12d0 <main+0x220>
  }
```

## Caveat

* Apart from strides 1, 2 and 4, the effect of vectorization/non-vectorization may not be straightforward to observe.
The overall performance can easily be memory-bound, rendering vectorization irrelevant--as CPU is not the bottleneck anyway.