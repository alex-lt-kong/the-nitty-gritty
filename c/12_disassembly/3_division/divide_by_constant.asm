for function divide_by_constant:


int result = a / 13;
<+0>:	movsxd rax,edi   ; cast 32-bit int a to 64-bit int.
<+3>:	sar    edi,0x1f  ; 0x1f == 31
; Shift Arithmetic Right. It is almost the same as SHR, except that most-significant bit (MSB) is shifted back to itself.
; This preserves the original sign of the destination operand, because MSB is the sign bit
<+6>:	imul   rax,rax,0x4ec4ec4f ; rax *= 1321528399
<+13>:	sar    rax,0x22           ; rax /= 2^34
; interestingly, 1321528399 / 2^34 == 1 / 13 == 0.07692307692307693
<+17>:	sub    eax,edi

return result;

12	}
<+19>:	ret