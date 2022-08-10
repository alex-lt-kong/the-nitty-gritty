for function floating_division_aos:
b, float c, struct pixel** arr, struct pixel** results, size_t arr_len) {
<+0>:	movaps xmm3,xmm0
<+3>:	mov    r8,rsi




are specific for the compiler and platform in use. So the best bet is to look at compiler's documentation.


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


33	}
<+80>:	ret    
nop WORD PTR cs:[rax+rax*1+0x0]
   DWORD PTR [rax+0x0]


for function floating_division_soa:
b, float c, struct pixelArray* arr, struct pixelArray* results, size_t arr_len) {
<+0>:	movaps xmm3,xmm0
<+3>:	mov    rcx,rdi
<+6>:	mov    rax,rsi


 

i = 0; i < arr_len; ++i) {
<+9>:	test   rdx,rdx
<+12>:	je     0xc5 <floating_division_soa+101>
<+92>:	add    rax,0x4
<+96>:	cmp    rdx,rax
<+99>:	jne    0x90 <floating_division_soa+48>

results->r[i] = arr->r[i] / a;
<+14>:	mov    r10,QWORD PTR [rdi]
<+17>:	mov    r9,QWORD PTR [rsi]
<+20>:	shl    rdx,0x2
<+48>:	movss  xmm0,DWORD PTR [r10+rax*1]
<+54>:	divss  xmm0,xmm3
<+58>:	movss  DWORD PTR [r9+rax*1],xmm0

results->g[i] = b / arr->g[i];
<+24>:	mov    r8,QWORD PTR [rdi+0x8]
<+28>:	mov    rdi,QWORD PTR [rsi+0x8]
<+64>:	movaps xmm0,xmm1
<+67>:	divss  xmm0,DWORD PTR [r8+rax*1]
<+73>:	movss  DWORD PTR [rdi+rax*1],xmm0

results->b[i] = arr->b[i] / c;
<+32>:	mov    rsi,QWORD PTR [rcx+0x10]
<+36>:	mov    rcx,QWORD PTR [rax+0x10]
<+40>:	xor    eax,eax
<+42>:	nop    WORD PTR [rax+rax*1+0x0]
<+78>:	movss  xmm0,DWORD PTR [rsi+rax*1]
<+83>:	divss  xmm0,xmm2
<+87>:	movss  DWORD PTR [rcx+rax*1],xmm0


44	}
<+101>:	ret    


