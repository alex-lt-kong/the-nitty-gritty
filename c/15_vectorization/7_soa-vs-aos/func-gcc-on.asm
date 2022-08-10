for function floating_division_aos:
b, float c, struct pixel** arr, struct pixel** results, size_t arr_len) {
<+0>:	movaps %xmm0,%xmm3
<+3>:	mov    %rsi,%r8




are specific for the compiler and platform in use. So the best bet is to look at compiler's documentation.


i = 0; i < arr_len; ++i) {
<+6>:	test   %rdx,%rdx
<+9>:	je     0x50 <floating_division_aos+80>
<+11>:	lea    0x0(,%rdx,8),%rsi
<+19>:	xor    %eax,%eax
<+21>:	nopl   (%rax)
<+32>:	add    $0x8,%rax
<+75>:	cmp    %rax,%rsi
<+78>:	jne    0x18 <floating_division_aos+24>

results[i]->r = arr[i]->r / a;
<+24>:	mov    (%rdi,%rax,1),%rcx
<+28>:	mov    (%r8,%rax,1),%rdx
<+36>:	movss  (%rcx),%xmm0
<+40>:	divss  %xmm3,%xmm0
<+44>:	movss  %xmm0,(%rdx)

results[i]->g = b / arr[i]->g;
<+48>:	movaps %xmm1,%xmm0
<+51>:	divss  0x4(%rcx),%xmm0
<+56>:	movss  %xmm0,0x4(%rdx)

results[i]->b = arr[i]->b / c;
<+61>:	movss  0x8(%rcx),%xmm0
<+66>:	divss  %xmm2,%xmm0
<+70>:	movss  %xmm0,0x8(%rdx)


33	}
<+80>:	ret    
nopw %cs:0x0(%rax,%rax,1)
  0x0(%rax)


for function floating_division_soa:
b, float c, struct pixelArray* arr, struct pixelArray* results, size_t arr_len) {
<+0>:	movaps %xmm0,%xmm3
<+3>:	mov    %rdi,%rcx
<+6>:	mov    %rsi,%rax


 

i = 0; i < arr_len; ++i) {
<+9>:	test   %rdx,%rdx
<+12>:	je     0xc5 <floating_division_soa+101>
<+92>:	add    $0x4,%rax
<+96>:	cmp    %rax,%rdx
<+99>:	jne    0x90 <floating_division_soa+48>

results->r[i] = arr->r[i] / a;
<+14>:	mov    (%rdi),%r10
<+17>:	mov    (%rsi),%r9
<+20>:	shl    $0x2,%rdx
<+48>:	movss  (%r10,%rax,1),%xmm0
<+54>:	divss  %xmm3,%xmm0
<+58>:	movss  %xmm0,(%r9,%rax,1)

results->g[i] = b / arr->g[i];
<+24>:	mov    0x8(%rdi),%r8
<+28>:	mov    0x8(%rsi),%rdi
<+64>:	movaps %xmm1,%xmm0
<+67>:	divss  (%r8,%rax,1),%xmm0
<+73>:	movss  %xmm0,(%rdi,%rax,1)

results->b[i] = arr->b[i] / c;
<+32>:	mov    0x10(%rcx),%rsi
<+36>:	mov    0x10(%rax),%rcx
<+40>:	xor    %eax,%eax
<+42>:	nopw   0x0(%rax,%rax,1)
<+78>:	movss  (%rsi,%rax,1),%xmm0
<+83>:	divss  %xmm2,%xmm0
<+87>:	movss  %xmm0,(%rcx,%rax,1)


44	}
<+101>:	ret    


