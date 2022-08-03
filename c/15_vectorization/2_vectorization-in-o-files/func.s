# mark_description "Intel(R) C Intel(R) 64 Compiler Classic for applications running on Intel(R) 64, Version 2021.6.0 Build 2022";
# mark_description "0226_000000";
# mark_description "-S -vec -qopt-report -qopt-report-phase=vec";
	.file "func.c"
	.text
..TXTST0:
.L_2__routine_start_linear_func_0:
# -- Begin  linear_func
	.text
# mark_begin;
       .align    16,0x90
	.globl linear_func
# --- linear_func(unsigned int *, unsigned int *, size_t)
linear_func:
# parameter 1: %rdi
# parameter 2: %rsi
# parameter 3: %rdx
..B1.1:                         # Preds ..B1.0
                                # Execution count [1.00e+00]
	.cfi_startproc
..___tag_value_linear_func.1:
..L2:
                                                          #6.78
        pushq     %r12                                          #6.78
	.cfi_def_cfa_offset 16
	.cfi_offset 12, -16
        pushq     %r13                                          #6.78
	.cfi_def_cfa_offset 24
	.cfi_offset 13, -24
        pushq     %r14                                          #6.78
	.cfi_def_cfa_offset 32
	.cfi_offset 14, -32
        pushq     %r15                                          #6.78
	.cfi_def_cfa_offset 40
	.cfi_offset 15, -40
        pushq     %rsi                                          #6.78
	.cfi_def_cfa_offset 48
        movq      %rdx, %r14                                    #6.78
        movq      %rsi, %r13                                    #6.78
        movq      %rdi, %r12                                    #6.78
#       rand(void)
        call      rand                                          #7.20
                                # LOE rbx rbp r12 r13 r14 eax
..B1.41:                        # Preds ..B1.1
                                # Execution count [1.00e+00]
        movl      %eax, %r15d                                   #7.20
                                # LOE rbx rbp r12 r13 r14 r15d
..B1.2:                         # Preds ..B1.41
                                # Execution count [1.00e+00]
        andl      $-2147482625, %r15d                           #7.29
        jge       ..B1.37       # Prob 50%                      #7.29
                                # LOE rbx rbp r12 r13 r14 r15d
..B1.38:                        # Preds ..B1.2
                                # Execution count [1.00e+00]
        subl      $1, %r15d                                     #7.29
        orl       $-1024, %r15d                                 #7.29
        incl      %r15d                                         #7.29
                                # LOE rbx rbp r12 r13 r14 r15d
..B1.37:                        # Preds ..B1.2 ..B1.38
                                # Execution count [1.00e+00]
#       rand(void)
        call      rand                                          #8.20
                                # LOE rbx rbp r12 r13 r14 eax r15d
..B1.42:                        # Preds ..B1.37
                                # Execution count [1.00e+00]
        movl      %eax, %ecx                                    #8.20
                                # LOE rbx rbp r12 r13 r14 ecx r15d
..B1.3:                         # Preds ..B1.42
                                # Execution count [1.00e+00]
        andl      $-2147482625, %ecx                            #8.29
        jge       ..B1.39       # Prob 50%                      #8.29
                                # LOE rbx rbp r12 r13 r14 ecx r15d
..B1.40:                        # Preds ..B1.3
                                # Execution count [1.00e+00]
        subl      $1, %ecx                                      #8.29
        orl       $-1024, %ecx                                  #8.29
        incl      %ecx                                          #8.29
                                # LOE rbx rbp r12 r13 r14 ecx r15d
..B1.39:                        # Preds ..B1.3 ..B1.40
                                # Execution count [1.00e+00]
        testq     %r14, %r14                                    #9.23
        jbe       ..B1.32       # Prob 50%                      #9.23
                                # LOE rbx rbp r12 r13 r14 ecx r15d
..B1.4:                         # Preds ..B1.39
                                # Execution count [0.00e+00]
        cmpq      $6, %r14                                      #9.3
        jbe       ..B1.26       # Prob 50%                      #9.3
                                # LOE rbx rbp r12 r13 r14 ecx r15d
..B1.5:                         # Preds ..B1.4
                                # Execution count [0.00e+00]
        movq      %r13, %rax                                    #10.22
        lea       (,%r14,4), %rdx                               #9.3
        subq      %r12, %rax                                    #10.22
        cmpq      %rdx, %rax                                    #9.3
        jge       ..B1.7        # Prob 50%                      #9.3
                                # LOE rax rdx rbx rbp r12 r13 r14 ecx r15d
..B1.6:                         # Preds ..B1.5
                                # Execution count [0.00e+00]
        negq      %rax                                          #10.5
        cmpq      %rdx, %rax                                    #9.3
        jl        ..B1.26       # Prob 50%                      #9.3
                                # LOE rbx rbp r12 r13 r14 ecx r15d
..B1.7:                         # Preds ..B1.6 ..B1.5
                                # Execution count [4.50e-01]
        movq      %r13, %rax                                    #9.3
        andq      $15, %rax                                     #9.3
        je        ..B1.10       # Prob 50%                      #9.3
                                # LOE rax rbx rbp r12 r13 r14 ecx r15d
..B1.8:                         # Preds ..B1.7
                                # Execution count [4.50e-01]
        testq     $3, %rax                                      #9.3
        jne       ..B1.33       # Prob 10%                      #9.3
                                # LOE rax rbx rbp r12 r13 r14 ecx r15d
..B1.9:                         # Preds ..B1.8
                                # Execution count [2.25e-01]
        negq      %rax                                          #9.3
        addq      $16, %rax                                     #9.3
        shrq      $2, %rax                                      #9.3
                                # LOE rax rbx rbp r12 r13 r14 ecx r15d
..B1.10:                        # Preds ..B1.9 ..B1.7
                                # Execution count [4.50e-01]
        lea       4(%rax), %rdx                                 #9.3
        cmpq      %rdx, %r14                                    #9.3
        jb        ..B1.33       # Prob 10%                      #9.3
                                # LOE rax rbx rbp r12 r13 r14 ecx r15d
..B1.11:                        # Preds ..B1.10
                                # Execution count [5.00e-01]
        movq      %r14, %rdx                                    #9.3
        xorl      %edi, %edi                                    #9.3
        subq      %rax, %rdx                                    #9.3
        xorl      %esi, %esi                                    #9.3
        andq      $3, %rdx                                      #9.3
        negq      %rdx                                          #9.3
        addq      %r14, %rdx                                    #9.3
        testq     %rax, %rax                                    #9.3
        jbe       ..B1.15       # Prob 9%                       #9.3
                                # LOE rax rdx rbx rbp rsi r12 r13 r14 ecx edi r15d
..B1.13:                        # Preds ..B1.11 ..B1.13
                                # Execution count [2.50e+00]
        movl      %r15d, %r8d                                   #10.22
        incl      %edi                                          #9.3
        imull     (%r12,%rsi,4), %r8d                           #10.22
        addl      %ecx, %r8d                                    #10.31
        movl      %r8d, (%r13,%rsi,4)                           #10.5
        incq      %rsi                                          #9.3
        cmpq      %rax, %rdi                                    #9.3
        jb        ..B1.13       # Prob 82%                      #9.3
                                # LOE rax rdx rbx rbp rsi r12 r13 r14 ecx edi r15d
..B1.15:                        # Preds ..B1.13 ..B1.11
                                # Execution count [2.50e+00]
        movl      %eax, %esi                                    #10.5
        lea       (%r12,%rax,4), %rdi                           #10.22
        testq     $15, %rdi                                     #9.3
        je        ..B1.19       # Prob 60%                      #9.3
                                # LOE rax rdx rbx rbp r12 r13 r14 ecx esi r15d
..B1.16:                        # Preds ..B1.15
                                # Execution count [4.50e-01]
        movd      %r15d, %xmm2                                  #7.18
        movd      %ecx, %xmm0                                   #8.18
        pshufd    $0, %xmm2, %xmm3                              #7.18
        movdqa    %xmm3, %xmm2                                  #10.22
        pshufd    $0, %xmm0, %xmm1                              #8.18
        psrlq     $32, %xmm2                                    #10.22
        movdqu    .L_2il0floatpacket.0(%rip), %xmm0             #10.22
                                # LOE rax rdx rbx rbp r12 r13 r14 ecx esi r15d xmm0 xmm1 xmm2 xmm3
..B1.17:                        # Preds ..B1.17 ..B1.16
                                # Execution count [2.50e+00]
        movdqu    (%r12,%rax,4), %xmm4                          #10.22
        movdqa    %xmm3, %xmm5                                  #10.22
        addl      $4, %esi                                      #9.3
        pmuludq   %xmm4, %xmm5                                  #10.22
        psrlq     $32, %xmm4                                    #10.22
        pmuludq   %xmm2, %xmm4                                  #10.22
        pand      %xmm0, %xmm5                                  #10.22
        psllq     $32, %xmm4                                    #10.22
        por       %xmm4, %xmm5                                  #10.22
        paddd     %xmm1, %xmm5                                  #10.31
        movdqu    %xmm5, (%r13,%rax,4)                          #10.5
        addq      $4, %rax                                      #9.3
        cmpq      %rdx, %rsi                                    #9.3
        jb        ..B1.17       # Prob 82%                      #9.3
        jmp       ..B1.22       # Prob 100%                     #9.3
                                # LOE rax rdx rbx rbp r12 r13 r14 ecx esi r15d xmm0 xmm1 xmm2 xmm3
..B1.19:                        # Preds ..B1.15
                                # Execution count [4.50e-01]
        movd      %r15d, %xmm2                                  #7.18
        movd      %ecx, %xmm0                                   #8.18
        pshufd    $0, %xmm2, %xmm3                              #7.18
        movdqa    %xmm3, %xmm2                                  #10.22
        pshufd    $0, %xmm0, %xmm1                              #8.18
        psrlq     $32, %xmm2                                    #10.22
        movdqu    .L_2il0floatpacket.0(%rip), %xmm0             #10.22
                                # LOE rax rdx rbx rbp r12 r13 r14 ecx esi r15d xmm0 xmm1 xmm2 xmm3
..B1.20:                        # Preds ..B1.20 ..B1.19
                                # Execution count [2.50e+00]
        movdqu    (%r12,%rax,4), %xmm4                          #10.22
        movdqa    %xmm3, %xmm5                                  #10.22
        pmuludq   %xmm4, %xmm5                                  #10.22
        psrlq     $32, %xmm4                                    #10.22
        pmuludq   %xmm2, %xmm4                                  #10.22
        pand      %xmm0, %xmm5                                  #10.22
        psllq     $32, %xmm4                                    #10.22
        por       %xmm4, %xmm5                                  #10.22
        addl      $4, %esi                                      #9.3
        paddd     %xmm1, %xmm5                                  #10.31
        movdqu    %xmm5, (%r13,%rax,4)                          #10.5
        addq      $4, %rax                                      #9.3
        cmpq      %rdx, %rsi                                    #9.3
        jb        ..B1.20       # Prob 82%                      #9.3
                                # LOE rax rdx rbx rbp r12 r13 r14 ecx esi r15d xmm0 xmm1 xmm2 xmm3
..B1.22:                        # Preds ..B1.20 ..B1.17 ..B1.33
                                # Execution count [5.00e-01]
        movslq    %edx, %rax                                    #9.3
        movl      %edx, %esi                                    #9.3
        movl      %edx, %edx                                    #9.3
        cmpq      %r14, %rdx                                    #9.3
        jae       ..B1.32       # Prob 9%                       #9.3
                                # LOE rax rbx rbp r12 r13 r14 ecx esi r15d
..B1.24:                        # Preds ..B1.22 ..B1.24
                                # Execution count [2.50e+00]
        movl      %r15d, %edx                                   #10.22
        incl      %esi                                          #9.3
        imull     (%r12,%rax,4), %edx                           #10.22
        addl      %ecx, %edx                                    #10.31
        movl      %edx, (%r13,%rax,4)                           #10.5
        incq      %rax                                          #9.3
        cmpq      %r14, %rsi                                    #9.3
        jb        ..B1.24       # Prob 82%                      #9.3
        jmp       ..B1.32       # Prob 100%                     #9.3
                                # LOE rax rbx rbp r12 r13 r14 ecx esi r15d
..B1.26:                        # Preds ..B1.6 ..B1.4
                                # Execution count [5.00e-01]
        movq      %r14, %rax                                    #6.6
        movl      $1, %esi                                      #9.3
        xorl      %edx, %edx                                    #9.3
        shrq      $1, %rax                                      #6.6
        je        ..B1.30       # Prob 9%                       #9.3
                                # LOE rax rbx rbp rsi r12 r13 r14 edx ecx r15d
..B1.28:                        # Preds ..B1.26 ..B1.28
                                # Execution count [1.25e+00]
        movslq    %edx, %rdx                                    #10.5
        movl      (%r12,%rdx,8), %esi                           #10.22
        imull     %r15d, %esi                                   #10.22
        addl      %ecx, %esi                                    #10.31
        movl      %esi, (%r13,%rdx,8)                           #10.5
        movl      4(%r12,%rdx,8), %edi                          #10.22
        imull     %r15d, %edi                                   #10.22
        addl      %ecx, %edi                                    #10.31
        movl      %edi, 4(%r13,%rdx,8)                          #10.5
        incl      %edx                                          #9.3
        cmpq      %rax, %rdx                                    #9.3
        jb        ..B1.28       # Prob 63%                      #9.3
                                # LOE rax rbx rbp r12 r13 r14 edx ecx r15d
..B1.29:                        # Preds ..B1.28
                                # Execution count [4.50e-01]
        addl      %edx, %edx                                    #9.3
        movslq    %edx, %rsi                                    #9.3
        incq      %rsi                                          #9.3
                                # LOE rbx rbp rsi r12 r13 r14 ecx r15d
..B1.30:                        # Preds ..B1.29 ..B1.26
                                # Execution count [5.00e-01]
        decq      %rsi                                          #9.3
        movl      %esi, %eax                                    #9.3
        cmpq      %r14, %rax                                    #9.3
        jae       ..B1.32       # Prob 9%                       #9.3
                                # LOE rbx rbp rsi r12 r13 ecx r15d
..B1.31:                        # Preds ..B1.30
                                # Execution count [4.50e-01]
        imull     (%r12,%rsi,4), %r15d                          #10.22
        addl      %r15d, %ecx                                   #10.31
        movl      %ecx, (%r13,%rsi,4)                           #10.5
                                # LOE rbx rbp
..B1.32:                        # Preds ..B1.24 ..B1.39 ..B1.30 ..B1.22 ..B1.31
                                #      
                                # Execution count [1.00e+00]
        popq      %rcx                                          #12.1
	.cfi_def_cfa_offset 40
	.cfi_restore 15
        popq      %r15                                          #12.1
	.cfi_def_cfa_offset 32
	.cfi_restore 14
        popq      %r14                                          #12.1
	.cfi_def_cfa_offset 24
	.cfi_restore 13
        popq      %r13                                          #12.1
	.cfi_def_cfa_offset 16
	.cfi_restore 12
        popq      %r12                                          #12.1
	.cfi_def_cfa_offset 8
        ret                                                     #12.1
	.cfi_def_cfa_offset 48
	.cfi_offset 12, -16
	.cfi_offset 13, -24
	.cfi_offset 14, -32
	.cfi_offset 15, -40
                                # LOE
..B1.33:                        # Preds ..B1.8 ..B1.10
                                # Execution count [4.50e-02]: Infreq
        xorl      %edx, %edx                                    #9.3
        jmp       ..B1.22       # Prob 100%                     #9.3
        .align    16,0x90
                                # LOE rdx rbx rbp r12 r13 r14 ecx r15d
	.cfi_endproc
# mark_end;
	.type	linear_func,@function
	.size	linear_func,.-linear_func
..LNlinear_func.0:
	.data
# -- End  linear_func
	.section .rodata, "a"
	.align 16
	.align 16
.L_2il0floatpacket.0:
	.long	0xffffffff,0x00000000,0xffffffff,0x00000000
	.type	.L_2il0floatpacket.0,@object
	.size	.L_2il0floatpacket.0,16
	.data
	.section .note.GNU-stack, ""
# End
