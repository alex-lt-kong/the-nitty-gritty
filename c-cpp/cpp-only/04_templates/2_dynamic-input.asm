
2_dynamic-input.out:     file format elf64-x86-64


Disassembly of section .init:

0000000000001000 <_init>:
    1000:	48 83 ec 08          	sub    rsp,0x8
    1004:	48 8b 05 dd 2f 00 00 	mov    rax,QWORD PTR [rip+0x2fdd]        # 3fe8 <__gmon_start__>
    100b:	48 85 c0             	test   rax,rax
    100e:	74 02                	je     1012 <_init+0x12>
    1010:	ff d0                	call   rax
    1012:	48 83 c4 08          	add    rsp,0x8
    1016:	c3                   	ret    

Disassembly of section .plt:

0000000000001020 <.plt>:
    1020:	ff 35 e2 2f 00 00    	push   QWORD PTR [rip+0x2fe2]        # 4008 <_GLOBAL_OFFSET_TABLE_+0x8>
    1026:	ff 25 e4 2f 00 00    	jmp    QWORD PTR [rip+0x2fe4]        # 4010 <_GLOBAL_OFFSET_TABLE_+0x10>
    102c:	0f 1f 40 00          	nop    DWORD PTR [rax+0x0]

0000000000001030 <rand@plt>:
    1030:	ff 25 e2 2f 00 00    	jmp    QWORD PTR [rip+0x2fe2]        # 4018 <rand@GLIBC_2.2.5>
    1036:	68 00 00 00 00       	push   0x0
    103b:	e9 e0 ff ff ff       	jmp    1020 <.plt>

0000000000001040 <std::basic_ostream<char, std::char_traits<char> >& std::endl<char, std::char_traits<char> >(std::basic_ostream<char, std::char_traits<char> >&)@plt>:
    1040:	ff 25 da 2f 00 00    	jmp    QWORD PTR [rip+0x2fda]        # 4020 <std::basic_ostream<char, std::char_traits<char> >& std::endl<char, std::char_traits<char> >(std::basic_ostream<char, std::char_traits<char> >&)@GLIBCXX_3.4>
    1046:	68 01 00 00 00       	push   0x1
    104b:	e9 d0 ff ff ff       	jmp    1020 <.plt>

0000000000001050 <__cxa_atexit@plt>:
    1050:	ff 25 d2 2f 00 00    	jmp    QWORD PTR [rip+0x2fd2]        # 4028 <__cxa_atexit@GLIBC_2.2.5>
    1056:	68 02 00 00 00       	push   0x2
    105b:	e9 c0 ff ff ff       	jmp    1020 <.plt>

0000000000001060 <time@plt>:
    1060:	ff 25 ca 2f 00 00    	jmp    QWORD PTR [rip+0x2fca]        # 4030 <time@GLIBC_2.2.5>
    1066:	68 03 00 00 00       	push   0x3
    106b:	e9 b0 ff ff ff       	jmp    1020 <.plt>

0000000000001070 <srand@plt>:
    1070:	ff 25 c2 2f 00 00    	jmp    QWORD PTR [rip+0x2fc2]        # 4038 <srand@GLIBC_2.2.5>
    1076:	68 04 00 00 00       	push   0x4
    107b:	e9 a0 ff ff ff       	jmp    1020 <.plt>

0000000000001080 <std::ios_base::Init::Init()@plt>:
    1080:	ff 25 ba 2f 00 00    	jmp    QWORD PTR [rip+0x2fba]        # 4040 <std::ios_base::Init::Init()@GLIBCXX_3.4>
    1086:	68 05 00 00 00       	push   0x5
    108b:	e9 90 ff ff ff       	jmp    1020 <.plt>

0000000000001090 <std::ostream& std::ostream::_M_insert<double>(double)@plt>:
    1090:	ff 25 b2 2f 00 00    	jmp    QWORD PTR [rip+0x2fb2]        # 4048 <std::ostream& std::ostream::_M_insert<double>(double)@GLIBCXX_3.4.9>
    1096:	68 06 00 00 00       	push   0x6
    109b:	e9 80 ff ff ff       	jmp    1020 <.plt>

00000000000010a0 <std::ostream::operator<<(int)@plt>:
    10a0:	ff 25 aa 2f 00 00    	jmp    QWORD PTR [rip+0x2faa]        # 4050 <std::ostream::operator<<(int)@GLIBCXX_3.4>
    10a6:	68 07 00 00 00       	push   0x7
    10ab:	e9 70 ff ff ff       	jmp    1020 <.plt>

Disassembly of section .plt.got:

00000000000010b0 <__cxa_finalize@plt>:
    10b0:	ff 25 1a 2f 00 00    	jmp    QWORD PTR [rip+0x2f1a]        # 3fd0 <__cxa_finalize@GLIBC_2.2.5>
    10b6:	66 90                	xchg   ax,ax

Disassembly of section .text:

00000000000010c0 <_start>:
    10c0:	31 ed                	xor    ebp,ebp
    10c2:	49 89 d1             	mov    r9,rdx
    10c5:	5e                   	pop    rsi
    10c6:	48 89 e2             	mov    rdx,rsp
    10c9:	48 83 e4 f0          	and    rsp,0xfffffffffffffff0
    10cd:	50                   	push   rax
    10ce:	54                   	push   rsp
    10cf:	4c 8d 05 6a 02 00 00 	lea    r8,[rip+0x26a]        # 1340 <__libc_csu_fini>
    10d6:	48 8d 0d 03 02 00 00 	lea    rcx,[rip+0x203]        # 12e0 <__libc_csu_init>
    10dd:	48 8d 3d fe 00 00 00 	lea    rdi,[rip+0xfe]        # 11e2 <main>
    10e4:	ff 15 f6 2e 00 00    	call   QWORD PTR [rip+0x2ef6]        # 3fe0 <__libc_start_main@GLIBC_2.2.5>
    10ea:	f4                   	hlt    
    10eb:	0f 1f 44 00 00       	nop    DWORD PTR [rax+rax*1+0x0]

00000000000010f0 <deregister_tm_clones>:
    10f0:	48 8d 3d 71 2f 00 00 	lea    rdi,[rip+0x2f71]        # 4068 <__TMC_END__>
    10f7:	48 8d 05 6a 2f 00 00 	lea    rax,[rip+0x2f6a]        # 4068 <__TMC_END__>
    10fe:	48 39 f8             	cmp    rax,rdi
    1101:	74 15                	je     1118 <deregister_tm_clones+0x28>
    1103:	48 8b 05 ce 2e 00 00 	mov    rax,QWORD PTR [rip+0x2ece]        # 3fd8 <_ITM_deregisterTMCloneTable>
    110a:	48 85 c0             	test   rax,rax
    110d:	74 09                	je     1118 <deregister_tm_clones+0x28>
    110f:	ff e0                	jmp    rax
    1111:	0f 1f 80 00 00 00 00 	nop    DWORD PTR [rax+0x0]
    1118:	c3                   	ret    
    1119:	0f 1f 80 00 00 00 00 	nop    DWORD PTR [rax+0x0]

0000000000001120 <register_tm_clones>:
    1120:	48 8d 3d 41 2f 00 00 	lea    rdi,[rip+0x2f41]        # 4068 <__TMC_END__>
    1127:	48 8d 35 3a 2f 00 00 	lea    rsi,[rip+0x2f3a]        # 4068 <__TMC_END__>
    112e:	48 29 fe             	sub    rsi,rdi
    1131:	48 89 f0             	mov    rax,rsi
    1134:	48 c1 ee 3f          	shr    rsi,0x3f
    1138:	48 c1 f8 03          	sar    rax,0x3
    113c:	48 01 c6             	add    rsi,rax
    113f:	48 d1 fe             	sar    rsi,1
    1142:	74 14                	je     1158 <register_tm_clones+0x38>
    1144:	48 8b 05 a5 2e 00 00 	mov    rax,QWORD PTR [rip+0x2ea5]        # 3ff0 <_ITM_registerTMCloneTable>
    114b:	48 85 c0             	test   rax,rax
    114e:	74 08                	je     1158 <register_tm_clones+0x38>
    1150:	ff e0                	jmp    rax
    1152:	66 0f 1f 44 00 00    	nop    WORD PTR [rax+rax*1+0x0]
    1158:	c3                   	ret    
    1159:	0f 1f 80 00 00 00 00 	nop    DWORD PTR [rax+0x0]

0000000000001160 <__do_global_dtors_aux>:
    1160:	80 3d 29 30 00 00 00 	cmp    BYTE PTR [rip+0x3029],0x0        # 4190 <completed.0>
    1167:	75 2f                	jne    1198 <__do_global_dtors_aux+0x38>
    1169:	55                   	push   rbp
    116a:	48 83 3d 5e 2e 00 00 	cmp    QWORD PTR [rip+0x2e5e],0x0        # 3fd0 <__cxa_finalize@GLIBC_2.2.5>
    1171:	00 
    1172:	48 89 e5             	mov    rbp,rsp
    1175:	74 0c                	je     1183 <__do_global_dtors_aux+0x23>
    1177:	48 8b 3d e2 2e 00 00 	mov    rdi,QWORD PTR [rip+0x2ee2]        # 4060 <__dso_handle>
    117e:	e8 2d ff ff ff       	call   10b0 <__cxa_finalize@plt>
    1183:	e8 68 ff ff ff       	call   10f0 <deregister_tm_clones>
    1188:	c6 05 01 30 00 00 01 	mov    BYTE PTR [rip+0x3001],0x1        # 4190 <completed.0>
    118f:	5d                   	pop    rbp
    1190:	c3                   	ret    
    1191:	0f 1f 80 00 00 00 00 	nop    DWORD PTR [rax+0x0]
    1198:	c3                   	ret    
    1199:	0f 1f 80 00 00 00 00 	nop    DWORD PTR [rax+0x0]

00000000000011a0 <frame_dummy>:
    11a0:	e9 7b ff ff ff       	jmp    1120 <register_tm_clones>

00000000000011a5 <__static_initialization_and_destruction_0(int, int)>:
    double a_dbl = (double)rand() / rand();
    double b_dbl = (double)rand() / rand();
    max_dbl = my_max(a_dbl, b_dbl);
    cout << max_dbl << endl;
    return 0;
    11a5:	83 ff 01             	cmp    edi,0x1
    11a8:	74 01                	je     11ab <__static_initialization_and_destruction_0(int, int)+0x6>
    11aa:	c3                   	ret    
    11ab:	81 fe ff ff 00 00    	cmp    esi,0xffff
    11b1:	75 f7                	jne    11aa <__static_initialization_and_destruction_0(int, int)+0x5>
    11b3:	48 83 ec 08          	sub    rsp,0x8
  extern wostream wclog;	/// Linked to standard error (buffered)
#endif
  //@}

  // For construction of filebuffers for cout, cin, cerr, clog et. al.
  static ios_base::Init __ioinit;
    11b7:	48 8d 3d d3 2f 00 00 	lea    rdi,[rip+0x2fd3]        # 4191 <std::__ioinit>
    11be:	e8 bd fe ff ff       	call   1080 <std::ios_base::Init::Init()@plt>
    11c3:	48 8d 15 96 2e 00 00 	lea    rdx,[rip+0x2e96]        # 4060 <__dso_handle>
    11ca:	48 8d 35 c0 2f 00 00 	lea    rsi,[rip+0x2fc0]        # 4191 <std::__ioinit>
    11d1:	48 8b 3d 20 2e 00 00 	mov    rdi,QWORD PTR [rip+0x2e20]        # 3ff8 <std::ios_base::Init::~Init()@GLIBCXX_3.4>
    11d8:	e8 73 fe ff ff       	call   1050 <__cxa_atexit@plt>
    11dd:	48 83 c4 08          	add    rsp,0x8
    11e1:	c3                   	ret    

00000000000011e2 <main>:
int main() {
    11e2:	53                   	push   rbx
    11e3:	48 83 ec 10          	sub    rsp,0x10
    srand(time(NULL));
    11e7:	bf 00 00 00 00       	mov    edi,0x0
    11ec:	e8 6f fe ff ff       	call   1060 <time@plt>
    11f1:	48 89 c7             	mov    rdi,rax
    11f4:	e8 77 fe ff ff       	call   1070 <srand@plt>
    int a_int = rand(), b_int = rand();
    11f9:	e8 32 fe ff ff       	call   1030 <rand@plt>
    11fe:	89 c3                	mov    ebx,eax
    1200:	e8 2b fe ff ff       	call   1030 <rand@plt>
    1205:	89 c6                	mov    esi,eax
    max_int = my_max(a_int, b_int);
    1207:	89 df                	mov    edi,ebx
    1209:	e8 b0 00 00 00       	call   12be <int my_max<int>(int, int)>
    120e:	89 c6                	mov    esi,eax
    cout << max_int << endl;
    1210:	48 8d 3d 69 2e 00 00 	lea    rdi,[rip+0x2e69]        # 4080 <std::cout@@GLIBCXX_3.4>
    1217:	e8 84 fe ff ff       	call   10a0 <std::ostream::operator<<(int)@plt>
    121c:	48 89 c7             	mov    rdi,rax
      operator<<(__ostream_type& (*__pf)(__ostream_type&))
      {
	// _GLIBCXX_RESOLVE_LIB_DEFECTS
	// DR 60. What is a formatted input function?
	// The inserters for manipulators are *not* formatted output functions.
	return __pf(*this);
    121f:	e8 1c fe ff ff       	call   1040 <std::basic_ostream<char, std::char_traits<char> >& std::endl<char, std::char_traits<char> >(std::basic_ostream<char, std::char_traits<char> >&)@plt>
    double a_dbl = (double)rand() / rand();
    1224:	e8 07 fe ff ff       	call   1030 <rand@plt>
    1229:	66 0f ef d2          	pxor   xmm2,xmm2
    122d:	f2 0f 2a d0          	cvtsi2sd xmm2,eax
    1231:	f2 0f 11 54 24 08    	movsd  QWORD PTR [rsp+0x8],xmm2
    1237:	e8 f4 fd ff ff       	call   1030 <rand@plt>
    123c:	66 0f ef c0          	pxor   xmm0,xmm0
    1240:	f2 0f 2a c0          	cvtsi2sd xmm0,eax
    1244:	f2 0f 10 54 24 08    	movsd  xmm2,QWORD PTR [rsp+0x8]
    124a:	f2 0f 5e d0          	divsd  xmm2,xmm0
    124e:	f2 0f 11 54 24 08    	movsd  QWORD PTR [rsp+0x8],xmm2
    double b_dbl = (double)rand() / rand();
    1254:	e8 d7 fd ff ff       	call   1030 <rand@plt>
    1259:	66 0f ef db          	pxor   xmm3,xmm3
    125d:	f2 0f 2a d8          	cvtsi2sd xmm3,eax
    1261:	66 48 0f 7e db       	movq   rbx,xmm3
    1266:	e8 c5 fd ff ff       	call   1030 <rand@plt>
    126b:	66 0f ef c0          	pxor   xmm0,xmm0
    126f:	f2 0f 2a c0          	cvtsi2sd xmm0,eax
    max_dbl = my_max(a_dbl, b_dbl);
    1273:	66 48 0f 6e cb       	movq   xmm1,rbx
    1278:	f2 0f 5e c8          	divsd  xmm1,xmm0
    127c:	f2 0f 10 44 24 08    	movsd  xmm0,QWORD PTR [rsp+0x8]
    1282:	e8 40 00 00 00       	call   12c7 <double my_max<double>(double, double)>
       *  These functions use the stream's current locale (specifically, the
       *  @c num_get facet) to perform numeric formatting.
      */
      __ostream_type&
      operator<<(double __f)
      { return _M_insert(__f); }
    1287:	48 8d 3d f2 2d 00 00 	lea    rdi,[rip+0x2df2]        # 4080 <std::cout@@GLIBCXX_3.4>
    128e:	e8 fd fd ff ff       	call   1090 <std::ostream& std::ostream::_M_insert<double>(double)@plt>
    1293:	48 89 c7             	mov    rdi,rax
	return __pf(*this);
    1296:	e8 a5 fd ff ff       	call   1040 <std::basic_ostream<char, std::char_traits<char> >& std::endl<char, std::char_traits<char> >(std::basic_ostream<char, std::char_traits<char> >&)@plt>
    129b:	b8 00 00 00 00       	mov    eax,0x0
    12a0:	48 83 c4 10          	add    rsp,0x10
    12a4:	5b                   	pop    rbx
    12a5:	c3                   	ret    

00000000000012a6 <_GLOBAL__sub_I_main>:
    12a6:	48 83 ec 08          	sub    rsp,0x8
    12aa:	be ff ff 00 00       	mov    esi,0xffff
    12af:	bf 01 00 00 00       	mov    edi,0x1
    12b4:	e8 ec fe ff ff       	call   11a5 <__static_initialization_and_destruction_0(int, int)>
    12b9:	48 83 c4 08          	add    rsp,0x8
    12bd:	c3                   	ret    

00000000000012be <int my_max<int>(int, int)>:
T my_max(T a, T b) {
    12be:	89 f0                	mov    eax,esi
    return a > b ? a : b;
    12c0:	39 f7                	cmp    edi,esi
    12c2:	7e 02                	jle    12c6 <int my_max<int>(int, int)+0x8>
    12c4:	89 f8                	mov    eax,edi
}
    12c6:	c3                   	ret    

00000000000012c7 <double my_max<double>(double, double)>:
    return a > b ? a : b;
    12c7:	66 0f 2f c1          	comisd xmm0,xmm1
    12cb:	76 04                	jbe    12d1 <double my_max<double>(double, double)+0xa>
    12cd:	66 0f 28 c8          	movapd xmm1,xmm0
}
    12d1:	66 0f 28 c1          	movapd xmm0,xmm1
    12d5:	c3                   	ret    
    12d6:	66 2e 0f 1f 84 00 00 	nop    WORD PTR cs:[rax+rax*1+0x0]
    12dd:	00 00 00 

00000000000012e0 <__libc_csu_init>:
    12e0:	41 57                	push   r15
    12e2:	4c 8d 3d df 2a 00 00 	lea    r15,[rip+0x2adf]        # 3dc8 <__frame_dummy_init_array_entry>
    12e9:	41 56                	push   r14
    12eb:	49 89 d6             	mov    r14,rdx
    12ee:	41 55                	push   r13
    12f0:	49 89 f5             	mov    r13,rsi
    12f3:	41 54                	push   r12
    12f5:	41 89 fc             	mov    r12d,edi
    12f8:	55                   	push   rbp
    12f9:	48 8d 2d d8 2a 00 00 	lea    rbp,[rip+0x2ad8]        # 3dd8 <__do_global_dtors_aux_fini_array_entry>
    1300:	53                   	push   rbx
    1301:	4c 29 fd             	sub    rbp,r15
    1304:	48 83 ec 08          	sub    rsp,0x8
    1308:	e8 f3 fc ff ff       	call   1000 <_init>
    130d:	48 c1 fd 03          	sar    rbp,0x3
    1311:	74 1b                	je     132e <__libc_csu_init+0x4e>
    1313:	31 db                	xor    ebx,ebx
    1315:	0f 1f 00             	nop    DWORD PTR [rax]
    1318:	4c 89 f2             	mov    rdx,r14
    131b:	4c 89 ee             	mov    rsi,r13
    131e:	44 89 e7             	mov    edi,r12d
    1321:	41 ff 14 df          	call   QWORD PTR [r15+rbx*8]
    1325:	48 83 c3 01          	add    rbx,0x1
    1329:	48 39 dd             	cmp    rbp,rbx
    132c:	75 ea                	jne    1318 <__libc_csu_init+0x38>
    132e:	48 83 c4 08          	add    rsp,0x8
    1332:	5b                   	pop    rbx
    1333:	5d                   	pop    rbp
    1334:	41 5c                	pop    r12
    1336:	41 5d                	pop    r13
    1338:	41 5e                	pop    r14
    133a:	41 5f                	pop    r15
    133c:	c3                   	ret    
    133d:	0f 1f 00             	nop    DWORD PTR [rax]

0000000000001340 <__libc_csu_fini>:
    1340:	c3                   	ret    

Disassembly of section .fini:

0000000000001344 <_fini>:
    1344:	48 83 ec 08          	sub    rsp,0x8
    1348:	48 83 c4 08          	add    rsp,0x8
    134c:	c3                   	ret    
