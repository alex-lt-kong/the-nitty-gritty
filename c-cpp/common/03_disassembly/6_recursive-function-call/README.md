  # Recursive function call

* gcc generates the machine code as follows:

```asm
0000000000000000 <factorial>:
   0:	push   rbp
   1:	mov    rbp,rsp
   4:	sub    rsp,0x10
   8:	mov    QWORD PTR [rbp-0x8],rdi
   c:	cmp    QWORD PTR [rbp-0x8],0x1
  11:	jne    19 <factorial+0x19>
  13:	mov    rax,QWORD PTR [rbp-0x8]
  17:	jmp    2e <factorial+0x2e>
  19:	mov    rax,QWORD PTR [rbp-0x8]
  1d:	sub    rax,0x1
  21:	mov    rdi,rax
  24:	call   29 <factorial+0x29>
  29:	imul   rax,QWORD PTR [rbp-0x8]
  2e:	leave  
  2f:	ret    
```

* We may want to decode it a bit more to follow what exactly are held in
registers:

```asm
0000000000000000 <factorial>:
   0:	push   rbp
   1:	mov    rbp,rsp
   4:	sub    rsp,0x10
   8:	mov    QWORD PTR [rbp-0x8],rdi
   c:	cmp    QWORD PTR [rbp-0x8],0x1
  11:	jne    19 <factorial+0x19>
  13:	mov    rax,QWORD PTR [rbp-0x8]
  17:	jmp    2e <factorial+0x2e>
  19:	mov    rax,QWORD PTR [rbp-0x8]
  1d:	sub    rax,0x1
  21:	mov    rdi,rax
  
  # === call ===
  24: sub    rsp, 8
  24: mov    [rsp], 29 
  24:	jmp    29 <factorial+0x29>

  29:	imul   rax,QWORD PTR [rbp-0x8]

  # === leave ===
  2e: mov    rsp, rbp 
  2e: mov    rbp, [rsp]
  2e: add    rsp, 8

  # === ret ===
  2f: mov    rcx, [rsp] # rcx is a general-purpose register we pick at random
  2f: add    rsp, 8  
  2f: jmp    rcx

  # Question: what is stores in rcx?
  # If line 24 is executed, it should just be 0, causing recursion
  # If `jne` at line 11 doesn't happen, we `jmp` to 2e directly,
  # skipping `call`ing at 24 , causing the call stack to unwind.
```
