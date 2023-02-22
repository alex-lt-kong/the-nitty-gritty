# Function call


* Calling a function is similar to (but slightly more than jmp to an address):

```
0000000000001125 <add>:
    1125:       55                      push   rbp
    1126:       48 89 e5                mov    rbp,rsp
    1129:       89 7d ec                mov    DWORD PTR [rbp-0x14],edi
    112c:       89 75 e8                mov    DWORD PTR [rbp-0x18],esi
    112f:       c7 45 fc 00 00 00 00    mov    DWORD PTR [rbp-0x4],0x0
    1136:       8b 55 ec                mov    edx,DWORD PTR [rbp-0x14]
    1139:       8b 45 e8                mov    eax,DWORD PTR [rbp-0x18]
    113c:       01 d0                   add    eax,edx
    113e:       89 45 fc                mov    DWORD PTR [rbp-0x4],eax
    1141:       8b 45 fc                mov    eax,DWORD PTR [rbp-0x4]
    1144:       5d                      pop    rbp
    1145:       c3                      ret

0000000000001146 <main>:
    1146:       55                      push   rbp
    1147:       48 89 e5                mov    rbp,rsp
    114a:       48 83 ec 10             sub    rsp,0x10
    114e:       c7 45 fc 03 00 00 00    mov    DWORD PTR [rbp-0x4],0x3
    1155:       c7 45 f8 39 30 00 00    mov    DWORD PTR [rbp-0x8],0x3039
    115c:       8b 55 f8                mov    edx,DWORD PTR [rbp-0x8]
    115f:       8b 45 fc                mov    eax,DWORD PTR [rbp-0x4]
    1162:       89 d6                   mov    esi,edx
    1164:       89 c7                   mov    edi,eax
    1166:       e8 ba ff ff ff          call   1125 <add>
    116b:       89 45 f4                mov    DWORD PTR [rbp-0xc],eax
    116e:       8b 45 f4                mov    eax,DWORD PTR [rbp-0xc]
    1171:       c9                      leave
    1172:       c3                      ret
    1173:       66 2e 0f 1f 84 00 00    nop    WORD PTR cs:[rax+rax*1+0x0]
    117a:       00 00 00
    117d:       0f 1f 00                nop    DWORD PTR [rax]
```

* Theoretically, it does the following two things:
    * It pushes the return address (address immediately after the CALL instruction) on the stack.
    * It changes `EIP` to the call destination. This effectively transfers control to the call target and begins execution there.