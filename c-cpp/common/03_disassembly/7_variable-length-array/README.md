# Variable-length array

* We know that the call stack is grown and variables are initialized in the new
call stack frame each time a function is called. But what happens if a
variable-length array (VLA) is involved, like this?
    ```C
    void print_vla(size_t arr_len) {
        int arr[arr_len];
        ...
        return;
    }
    ```

* The handling is actually rather straightforward--the call stack will grow
by a variable that is determined at runtime:
    ```asm
    0:	push   rbp
    1:	mov    rbp,rsp
    4:	push   rbx
    5:	sub    rsp,0x38                 # reserved 0x38 bytes for the new stack frame
    9:	mov    QWORD PTR [rbp-0x38],rdi # rdi stores the first parameter, which is arr_len
    ...
    13:	mov    rax,QWORD PTR [rbp-0x38]
    ...
    55:	imul   rax,rax,0x10
    59:	sub    rsp,rax                  # rsp, storing the "top" of stack, is moved again, allocating space for the VLA.
    ```

* Note that VLA is one of a few features that are valid in C but do not valid
in C++.