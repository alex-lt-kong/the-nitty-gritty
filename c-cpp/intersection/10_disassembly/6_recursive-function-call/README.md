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

* We may want to decode it a bit more to follow what exactly is held in
registers:

```asm
0000000000000000 <factorial>:
  # === push ===
   0: sub    rsp, 8
   0: mov    [rsp], rbp 

   1:	mov    rbp,rsp
   4:	sub    rsp,0x10
   8:	mov    QWORD PTR [rbp-0x8],rdi # only value, a
   c:	cmp    QWORD PTR [rbp-0x8],0x1
  11:	jne    19 <factorial+0x19>     # jump if not equal
  13:	mov    rax,QWORD PTR [rbp-0x8]
  17:	jmp    2e <factorial+0x2e>
  19:	mov    rax,QWORD PTR [rbp-0x8]
  1d:	sub    rax,0x1                 # --a;
  21:	mov    rdi,rax
  
  # === call ===
  24: sub    rsp, 8
  24: mov    [rsp], 29 
  24:	jmp    29 <factorial+0x29>

  29:	imul   rax,QWORD PTR [rbp-0x8] # rax is a-1, [rbp-0x8] is still a

  # === leave ===
  # === leave.mov ===                                     
  2e: mov    rsp, rbp                # rbp remains unchanged since 1, so here we restore rsp's original value at line 1
  # === leave.pop ===
  2e: mov    rbp, [rsp]              # we restore rbp's value as at line 0
  2e: add    rsp, 8                  # at line 0, rsp -= 8, now we reverse, so rsp is the same as its value at line 0.

  # === ret ===                      # rcx is a general-purpose register we pick at random
  2f: mov    rcx, [rsp]              # rcx holds rsp's value at line 0, i.e., rbp.
  2f: add    rsp, 8                  # This is to offset line 0's sub rsp, 8
  2f: jmp    [rcx]                   # jmp back to *rbp

  # Question: what is stores in rcx?
  # If line 24 is executed, it should just be 0, causing recursion
  # If `jne` at line 11 doesn't happen, we `jmp` to 2e directly,
  # skipping `call`ing at 24 , causing the call stack to unwind.
```


