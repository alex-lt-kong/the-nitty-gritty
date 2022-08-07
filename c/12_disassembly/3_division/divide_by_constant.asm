for function divide_by_constant:


int result = a / 271;
<+0>:	movsxd rax,edi
<+3>:	sar    edi,0x1f
<+6>:	imul   rax,rax,0xf1d48bd
<+13>:	sar    rax,0x24
<+17>:	sub    eax,edi

return result;

12	}
<+19>:	ret