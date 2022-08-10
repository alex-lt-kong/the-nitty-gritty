for function floating_division_aos:
b, float c, struct pixel** arr, struct pixel** results, size_t arr_len) {
<+0>:	movaps xmm3,xmm0
<+3>:	mov    r8,rsi


__INTEL_COMPILER)

are specific for the compiler and platform in use. So the best bet is to look at compiler's documentation.


ivdep

i = 0; i < arr_len; ++i) {
<+6>:	test   rdx,rdx
<+9>:	je     0x50 <floating_division_aos+80>
<+11>:	lea    rsi,[rdx*8+0x0]
<+19>:	xor    eax,eax
<+21>:	nop    DWORD PTR [rax]
<+32>:	add    rax,0x8
<+75>:	cmp    rsi,rax
<+78>:	jne    0x18 <floating_division_aos+24>

results[i]->r = arr[i]->r / a;
<+24>:	mov    rcx,QWORD PTR [rdi+rax*1]
<+28>:	mov    rdx,QWORD PTR [r8+rax*1]
<+36>:	movss  xmm0,DWORD PTR [rcx]
<+40>:	divss  xmm0,xmm3
<+44>:	movss  DWORD PTR [rdx],xmm0

results[i]->g = b / arr[i]->g;
<+48>:	movaps xmm0,xmm1
<+51>:	divss  xmm0,DWORD PTR [rcx+0x4]
<+56>:	movss  DWORD PTR [rdx+0x4],xmm0

results[i]->b = arr[i]->b / c;
<+61>:	movss  xmm0,DWORD PTR [rcx+0x8]
<+66>:	divss  xmm0,xmm2
<+70>:	movss  DWORD PTR [rdx+0x8],xmm0


35	}
<+80>:	ret    
nop WORD PTR cs:[rax+rax*1+0x0]
   DWORD PTR [rax+0x0]


for function floating_division_soa:
b, float c, struct pixelArray* arr, struct pixelArray* results, size_t arr_len) {
<+0>:	movaps xmm6,xmm0
<+3>:	mov    rcx,rdi
<+6>:	mov    rax,rsi

__INTEL_COMPILER)


ivdep

i = 0; i < arr_len; ++i) {
<+9>:	test   rdx,rdx
<+12>:	je     0x1a0 <floating_division_soa+320>
<+135>:	add    rax,0x10
<+139>:	cmp    rax,r11
<+142>:	jne    0xc0 <floating_division_soa+96>
<+144>:	mov    r11,rdx
<+147>:	and    r11,0xfffffffffffffffc
<+151>:	mov    eax,r11d
<+154>:	cmp    rdx,r11
<+157>:	je     0x1a8 <floating_division_soa+328>
<+210>:	lea    r11d,[rax+0x1]
<+214>:	movsxd r11,r11d
<+217>:	cmp    rdx,r11
<+220>:	jbe    0x1a0 <floating_division_soa+320>
<+228>:	add    eax,0x2
<+231>:	cdqe   
<+274>:	cmp    rdx,rax
<+277>:	jbe    0x1a0 <floating_division_soa+320>
<+329>:	xor    eax,eax
<+331>:	xor    r11d,r11d
<+334>:	jmp    0x103 <floating_division_soa+163>

results->r[i] = arr->r[i] / a;
<+18>:	mov    r10,QWORD PTR [rdi]
<+21>:	mov    r9,QWORD PTR [rsi]
<+96>:	movups xmm0,XMMWORD PTR [r10+rax*1]
<+101>:	divps  xmm0,xmm5
<+104>:	movups XMMWORD PTR [r9+rax*1],xmm0
<+163>:	movss  xmm0,DWORD PTR [r10+r11*4]
<+169>:	divss  xmm0,xmm6
<+173>:	movss  DWORD PTR [r9+r11*4],xmm0
<+222>:	movss  xmm0,DWORD PTR [r10+r11*4]
<+233>:	divss  xmm0,xmm6
<+237>:	movss  DWORD PTR [r9+r11*4],xmm0
<+279>:	movss  xmm0,DWORD PTR [r10+rax*4]
<+285>:	divss  xmm0,xmm6
<+289>:	movss  DWORD PTR [r9+rax*4],xmm0

results->g[i] = b / arr->g[i];
<+24>:	mov    r8,QWORD PTR [rdi+0x8]
<+28>:	mov    rdi,QWORD PTR [rsi+0x8]
<+109>:	movups xmm7,XMMWORD PTR [r8+rax*1]
<+114>:	movaps xmm0,xmm4
<+117>:	divps  xmm0,xmm7
<+120>:	movups XMMWORD PTR [rdi+rax*1],xmm0
<+179>:	movaps xmm0,xmm1
<+182>:	divss  xmm0,DWORD PTR [r8+r11*4]
<+188>:	movss  DWORD PTR [rdi+r11*4],xmm0
<+243>:	movaps xmm0,xmm1
<+246>:	divss  xmm0,DWORD PTR [r8+r11*4]
<+252>:	movss  DWORD PTR [rdi+r11*4],xmm0
<+295>:	divss  xmm1,DWORD PTR [r8+rax*4]
<+301>:	movss  DWORD PTR [rdi+rax*4],xmm1

results->b[i] = arr->b[i] / c;
<+32>:	mov    rsi,QWORD PTR [rcx+0x10]
<+36>:	mov    rcx,QWORD PTR [rax+0x10]
<+40>:	lea    rax,[rdx-0x1]
<+44>:	cmp    rax,0x2
<+48>:	jbe    0x1a9 <floating_division_soa+329>
<+54>:	mov    r11,rdx
<+57>:	movaps xmm5,xmm0
<+60>:	movaps xmm4,xmm1
<+63>:	xor    eax,eax
<+65>:	shr    r11,0x2
<+69>:	movaps xmm3,xmm2
<+72>:	shufps xmm5,xmm5,0x0
<+76>:	shufps xmm4,xmm4,0x0
<+80>:	shl    r11,0x4
<+84>:	shufps xmm3,xmm3,0x0
<+88>:	nop    DWORD PTR [rax+rax*1+0x0]
<+124>:	movups xmm0,XMMWORD PTR [rsi+rax*1]
<+128>:	divps  xmm0,xmm3
<+131>:	movups XMMWORD PTR [rcx+rax*1],xmm0
<+194>:	movss  xmm0,DWORD PTR [rsi+r11*4]
<+200>:	divss  xmm0,xmm2
<+204>:	movss  DWORD PTR [rcx+r11*4],xmm0
<+258>:	movss  xmm0,DWORD PTR [rsi+r11*4]
<+264>:	divss  xmm0,xmm2
<+268>:	movss  DWORD PTR [rcx+r11*4],xmm0
<+306>:	movss  xmm0,DWORD PTR [rsi+rax*4]
<+311>:	divss  xmm0,xmm2
<+315>:	movss  DWORD PTR [rcx+rax*4],xmm0


48	}
<+320>:	ret    
<+321>:	nop    DWORD PTR [rax+0x0]
<+328>:	ret    


