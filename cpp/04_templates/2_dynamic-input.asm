
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

0000000000001030 <std::ostream::put(char)@plt>:
    1030:	ff 25 e2 2f 00 00    	jmp    QWORD PTR [rip+0x2fe2]        # 4018 <std::ostream::put(char)@GLIBCXX_3.4>
    1036:	68 00 00 00 00       	push   0x0
    103b:	e9 e0 ff ff ff       	jmp    1020 <.plt>

0000000000001040 <rand@plt>:
    1040:	ff 25 da 2f 00 00    	jmp    QWORD PTR [rip+0x2fda]        # 4020 <rand@GLIBC_2.2.5>
    1046:	68 01 00 00 00       	push   0x1
    104b:	e9 d0 ff ff ff       	jmp    1020 <.plt>

0000000000001050 <std::ostream::flush()@plt>:
    1050:	ff 25 d2 2f 00 00    	jmp    QWORD PTR [rip+0x2fd2]        # 4028 <std::ostream::flush()@GLIBCXX_3.4>
    1056:	68 02 00 00 00       	push   0x2
    105b:	e9 c0 ff ff ff       	jmp    1020 <.plt>

0000000000001060 <__cxa_atexit@plt>:
    1060:	ff 25 ca 2f 00 00    	jmp    QWORD PTR [rip+0x2fca]        # 4030 <__cxa_atexit@GLIBC_2.2.5>
    1066:	68 03 00 00 00       	push   0x3
    106b:	e9 b0 ff ff ff       	jmp    1020 <.plt>

0000000000001070 <time@plt>:
    1070:	ff 25 c2 2f 00 00    	jmp    QWORD PTR [rip+0x2fc2]        # 4038 <time@GLIBC_2.2.5>
    1076:	68 04 00 00 00       	push   0x4
    107b:	e9 a0 ff ff ff       	jmp    1020 <.plt>

0000000000001080 <srand@plt>:
    1080:	ff 25 ba 2f 00 00    	jmp    QWORD PTR [rip+0x2fba]        # 4040 <srand@GLIBC_2.2.5>
    1086:	68 05 00 00 00       	push   0x5
    108b:	e9 90 ff ff ff       	jmp    1020 <.plt>

0000000000001090 <std::ctype<char>::_M_widen_init() const@plt>:
    1090:	ff 25 b2 2f 00 00    	jmp    QWORD PTR [rip+0x2fb2]        # 4048 <std::ctype<char>::_M_widen_init() const@GLIBCXX_3.4.11>
    1096:	68 06 00 00 00       	push   0x6
    109b:	e9 80 ff ff ff       	jmp    1020 <.plt>

00000000000010a0 <std::__throw_bad_cast()@plt>:
    10a0:	ff 25 aa 2f 00 00    	jmp    QWORD PTR [rip+0x2faa]        # 4050 <std::__throw_bad_cast()@GLIBCXX_3.4>
    10a6:	68 07 00 00 00       	push   0x7
    10ab:	e9 70 ff ff ff       	jmp    1020 <.plt>

00000000000010b0 <std::ios_base::Init::Init()@plt>:
    10b0:	ff 25 a2 2f 00 00    	jmp    QWORD PTR [rip+0x2fa2]        # 4058 <std::ios_base::Init::Init()@GLIBCXX_3.4>
    10b6:	68 08 00 00 00       	push   0x8
    10bb:	e9 60 ff ff ff       	jmp    1020 <.plt>

00000000000010c0 <std::ostream& std::ostream::_M_insert<double>(double)@plt>:
    10c0:	ff 25 9a 2f 00 00    	jmp    QWORD PTR [rip+0x2f9a]        # 4060 <std::ostream& std::ostream::_M_insert<double>(double)@GLIBCXX_3.4.9>
    10c6:	68 09 00 00 00       	push   0x9
    10cb:	e9 50 ff ff ff       	jmp    1020 <.plt>

00000000000010d0 <std::ostream::operator<<(int)@plt>:
    10d0:	ff 25 92 2f 00 00    	jmp    QWORD PTR [rip+0x2f92]        # 4068 <std::ostream::operator<<(int)@GLIBCXX_3.4>
    10d6:	68 0a 00 00 00       	push   0xa
    10db:	e9 40 ff ff ff       	jmp    1020 <.plt>

Disassembly of section .plt.got:

00000000000010e0 <__cxa_finalize@plt>:
    10e0:	ff 25 ea 2e 00 00    	jmp    QWORD PTR [rip+0x2eea]        # 3fd0 <__cxa_finalize@GLIBC_2.2.5>
    10e6:	66 90                	xchg   ax,ax

Disassembly of section .text:

00000000000010f0 <main>:
template<typename T>
T my_max(T a, T b) {
    return a > b ? a : b;
}

int main() {
    10f0:	53                   	push   rbx
    srand(time(NULL));
    10f1:	31 ff                	xor    edi,edi
int main() {
    10f3:	48 83 ec 10          	sub    rsp,0x10
    srand(time(NULL));
    10f7:	e8 74 ff ff ff       	call   1070 <time@plt>
    10fc:	48 89 c7             	mov    rdi,rax
    10ff:	e8 7c ff ff ff       	call   1080 <srand@plt>

    int max_int;
    int a_int = rand(), b_int = rand();
    1104:	e8 37 ff ff ff       	call   1040 <rand@plt>
    1109:	89 c3                	mov    ebx,eax
    110b:	e8 30 ff ff ff       	call   1040 <rand@plt>
    max_int = my_max(a_int, b_int);
    cout << max_int << endl;
    1110:	48 8d 3d 69 2f 00 00 	lea    rdi,[rip+0x2f69]        # 4080 <std::cout@@GLIBCXX_3.4>
    1117:	39 c3                	cmp    ebx,eax
    1119:	0f 4d c3             	cmovge eax,ebx
    111c:	89 c6                	mov    esi,eax
    111e:	e8 ad ff ff ff       	call   10d0 <std::ostream::operator<<(int)@plt>
    1123:	48 89 c7             	mov    rdi,rax
      operator<<(__ostream_type& (*__pf)(__ostream_type&))
      {
	// _GLIBCXX_RESOLVE_LIB_DEFECTS
	// DR 60. What is a formatted input function?
	// The inserters for manipulators are *not* formatted output functions.
	return __pf(*this);
    1126:	e8 95 01 00 00       	call   12c0 <std::basic_ostream<char, std::char_traits<char> >& std::endl<char, std::char_traits<char> >(std::basic_ostream<char, std::char_traits<char> >&) [clone .isra.0]>
    
    double max_dbl;
    double a_dbl = (double)rand() / rand();
    112b:	e8 10 ff ff ff       	call   1040 <rand@plt>
    1130:	89 c3                	mov    ebx,eax
    1132:	e8 09 ff ff ff       	call   1040 <rand@plt>
    1137:	66 0f ef c0          	pxor   xmm0,xmm0
    113b:	66 0f ef c9          	pxor   xmm1,xmm1
    113f:	f2 0f 2a c8          	cvtsi2sd xmm1,eax
    1143:	f2 0f 2a c3          	cvtsi2sd xmm0,ebx
    1147:	f2 0f 5e c1          	divsd  xmm0,xmm1
    114b:	f2 0f 11 44 24 08    	movsd  QWORD PTR [rsp+0x8],xmm0
    double b_dbl = (double)rand() / rand();
    1151:	e8 ea fe ff ff       	call   1040 <rand@plt>
    1156:	89 c3                	mov    ebx,eax
    1158:	e8 e3 fe ff ff       	call   1040 <rand@plt>
    115d:	66 0f ef c9          	pxor   xmm1,xmm1
    1161:	66 0f ef d2          	pxor   xmm2,xmm2
    return a > b ? a : b;
    1165:	f2 0f 10 44 24 08    	movsd  xmm0,QWORD PTR [rsp+0x8]
    double b_dbl = (double)rand() / rand();
    116b:	f2 0f 2a d0          	cvtsi2sd xmm2,eax
       *  These functions use the stream's current locale (specifically, the
       *  @c num_get facet) to perform numeric formatting.
      */
      __ostream_type&
      operator<<(double __f)
      { return _M_insert(__f); }
    116f:	48 8d 3d 0a 2f 00 00 	lea    rdi,[rip+0x2f0a]        # 4080 <std::cout@@GLIBCXX_3.4>
    1176:	f2 0f 2a cb          	cvtsi2sd xmm1,ebx
    117a:	f2 0f 5e ca          	divsd  xmm1,xmm2
    return a > b ? a : b;
    117e:	f2 0f 5f c1          	maxsd  xmm0,xmm1
    1182:	e8 39 ff ff ff       	call   10c0 <std::ostream& std::ostream::_M_insert<double>(double)@plt>
    1187:	48 89 c7             	mov    rdi,rax
	return __pf(*this);
    118a:	e8 31 01 00 00       	call   12c0 <std::basic_ostream<char, std::char_traits<char> >& std::endl<char, std::char_traits<char> >(std::basic_ostream<char, std::char_traits<char> >&) [clone .isra.0]>
    max_dbl = my_max(a_dbl, b_dbl);
    cout << max_dbl << endl;
    return 0;
    118f:	48 83 c4 10          	add    rsp,0x10
    1193:	31 c0                	xor    eax,eax
    1195:	5b                   	pop    rbx
    1196:	c3                   	ret    
    1197:	66 0f 1f 84 00 00 00 	nop    WORD PTR [rax+rax*1+0x0]
    119e:	00 00 

00000000000011a0 <_GLOBAL__sub_I_main>:
    11a0:	48 83 ec 08          	sub    rsp,0x8
  extern wostream wclog;	/// Linked to standard error (buffered)
#endif
  //@}

  // For construction of filebuffers for cout, cin, cerr, clog et. al.
  static ios_base::Init __ioinit;
    11a4:	48 8d 3d e6 2f 00 00 	lea    rdi,[rip+0x2fe6]        # 4191 <std::__ioinit>
    11ab:	e8 00 ff ff ff       	call   10b0 <std::ios_base::Init::Init()@plt>
    11b0:	48 8b 3d 41 2e 00 00 	mov    rdi,QWORD PTR [rip+0x2e41]        # 3ff8 <std::ios_base::Init::~Init()@GLIBCXX_3.4>
    11b7:	48 83 c4 08          	add    rsp,0x8
    11bb:	48 8d 15 b6 2e 00 00 	lea    rdx,[rip+0x2eb6]        # 4078 <__dso_handle>
    11c2:	48 8d 35 c8 2f 00 00 	lea    rsi,[rip+0x2fc8]        # 4191 <std::__ioinit>
    11c9:	e9 92 fe ff ff       	jmp    1060 <__cxa_atexit@plt>
    11ce:	66 90                	xchg   ax,ax

00000000000011d0 <_start>:
    11d0:	31 ed                	xor    ebp,ebp
    11d2:	49 89 d1             	mov    r9,rdx
    11d5:	5e                   	pop    rsi
    11d6:	48 89 e2             	mov    rdx,rsp
    11d9:	48 83 e4 f0          	and    rsp,0xfffffffffffffff0
    11dd:	50                   	push   rax
    11de:	54                   	push   rsp
    11df:	4c 8d 05 aa 01 00 00 	lea    r8,[rip+0x1aa]        # 1390 <__libc_csu_fini>
    11e6:	48 8d 0d 43 01 00 00 	lea    rcx,[rip+0x143]        # 1330 <__libc_csu_init>
    11ed:	48 8d 3d fc fe ff ff 	lea    rdi,[rip+0xfffffffffffffefc]        # 10f0 <main>
    11f4:	ff 15 e6 2d 00 00    	call   QWORD PTR [rip+0x2de6]        # 3fe0 <__libc_start_main@GLIBC_2.2.5>
    11fa:	f4                   	hlt    
    11fb:	0f 1f 44 00 00       	nop    DWORD PTR [rax+rax*1+0x0]

0000000000001200 <deregister_tm_clones>:
    1200:	48 8d 3d 79 2e 00 00 	lea    rdi,[rip+0x2e79]        # 4080 <std::cout@@GLIBCXX_3.4>
    1207:	48 8d 05 72 2e 00 00 	lea    rax,[rip+0x2e72]        # 4080 <std::cout@@GLIBCXX_3.4>
    120e:	48 39 f8             	cmp    rax,rdi
    1211:	74 15                	je     1228 <deregister_tm_clones+0x28>
    1213:	48 8b 05 be 2d 00 00 	mov    rax,QWORD PTR [rip+0x2dbe]        # 3fd8 <_ITM_deregisterTMCloneTable>
    121a:	48 85 c0             	test   rax,rax
    121d:	74 09                	je     1228 <deregister_tm_clones+0x28>
    121f:	ff e0                	jmp    rax
    1221:	0f 1f 80 00 00 00 00 	nop    DWORD PTR [rax+0x0]
    1228:	c3                   	ret    
    1229:	0f 1f 80 00 00 00 00 	nop    DWORD PTR [rax+0x0]

0000000000001230 <register_tm_clones>:
    1230:	48 8d 3d 49 2e 00 00 	lea    rdi,[rip+0x2e49]        # 4080 <std::cout@@GLIBCXX_3.4>
    1237:	48 8d 35 42 2e 00 00 	lea    rsi,[rip+0x2e42]        # 4080 <std::cout@@GLIBCXX_3.4>
    123e:	48 29 fe             	sub    rsi,rdi
    1241:	48 89 f0             	mov    rax,rsi
    1244:	48 c1 ee 3f          	shr    rsi,0x3f
    1248:	48 c1 f8 03          	sar    rax,0x3
    124c:	48 01 c6             	add    rsi,rax
    124f:	48 d1 fe             	sar    rsi,1
    1252:	74 14                	je     1268 <register_tm_clones+0x38>
    1254:	48 8b 05 95 2d 00 00 	mov    rax,QWORD PTR [rip+0x2d95]        # 3ff0 <_ITM_registerTMCloneTable>
    125b:	48 85 c0             	test   rax,rax
    125e:	74 08                	je     1268 <register_tm_clones+0x38>
    1260:	ff e0                	jmp    rax
    1262:	66 0f 1f 44 00 00    	nop    WORD PTR [rax+rax*1+0x0]
    1268:	c3                   	ret    
    1269:	0f 1f 80 00 00 00 00 	nop    DWORD PTR [rax+0x0]

0000000000001270 <__do_global_dtors_aux>:
    1270:	80 3d 19 2f 00 00 00 	cmp    BYTE PTR [rip+0x2f19],0x0        # 4190 <completed.0>
    1277:	75 2f                	jne    12a8 <__do_global_dtors_aux+0x38>
    1279:	55                   	push   rbp
    127a:	48 83 3d 4e 2d 00 00 	cmp    QWORD PTR [rip+0x2d4e],0x0        # 3fd0 <__cxa_finalize@GLIBC_2.2.5>
    1281:	00 
    1282:	48 89 e5             	mov    rbp,rsp
    1285:	74 0c                	je     1293 <__do_global_dtors_aux+0x23>
    1287:	48 8b 3d ea 2d 00 00 	mov    rdi,QWORD PTR [rip+0x2dea]        # 4078 <__dso_handle>
    128e:	e8 4d fe ff ff       	call   10e0 <__cxa_finalize@plt>
    1293:	e8 68 ff ff ff       	call   1200 <deregister_tm_clones>
    1298:	c6 05 f1 2e 00 00 01 	mov    BYTE PTR [rip+0x2ef1],0x1        # 4190 <completed.0>
    129f:	5d                   	pop    rbp
    12a0:	c3                   	ret    
    12a1:	0f 1f 80 00 00 00 00 	nop    DWORD PTR [rax+0x0]
    12a8:	c3                   	ret    
    12a9:	0f 1f 80 00 00 00 00 	nop    DWORD PTR [rax+0x0]

00000000000012b0 <frame_dummy>:
    12b0:	e9 7b ff ff ff       	jmp    1230 <register_tm_clones>
    12b5:	66 2e 0f 1f 84 00 00 	nop    WORD PTR cs:[rax+rax*1+0x0]
    12bc:	00 00 00 
    12bf:	90                   	nop

00000000000012c0 <std::basic_ostream<char, std::char_traits<char> >& std::endl<char, std::char_traits<char> >(std::basic_ostream<char, std::char_traits<char> >&) [clone .isra.0]>:
   *  https://gcc.gnu.org/onlinedocs/libstdc++/manual/streambufs.html#io.streambuf.buffering
   *  for more on this subject.
  */
  template<typename _CharT, typename _Traits>
    inline basic_ostream<_CharT, _Traits>&
    endl(basic_ostream<_CharT, _Traits>& __os)
    12c0:	41 54                	push   r12
    12c2:	55                   	push   rbp
    12c3:	48 83 ec 08          	sub    rsp,0x8
    { return flush(__os.put(__os.widen('\n'))); }
    12c7:	48 8b 07             	mov    rax,QWORD PTR [rdi]
    12ca:	48 8b 40 e8          	mov    rax,QWORD PTR [rax-0x18]
    12ce:	4c 8b a4 07 f0 00 00 	mov    r12,QWORD PTR [rdi+rax*1+0xf0]
    12d5:	00 

  template<typename _Facet>
    inline const _Facet&
    __check_facet(const _Facet* __f)
    {
      if (!__f)
    12d6:	4d 85 e4             	test   r12,r12
    12d9:	74 44                	je     131f <std::basic_ostream<char, std::char_traits<char> >& std::endl<char, std::char_traits<char> >(std::basic_ostream<char, std::char_traits<char> >&) [clone .isra.0]+0x5f>
       *  @return  The converted character.
      */
      char_type
      widen(char __c) const
      {
	if (_M_widen_ok)
    12db:	41 80 7c 24 38 00    	cmp    BYTE PTR [r12+0x38],0x0
    12e1:	48 89 fd             	mov    rbp,rdi
    12e4:	74 1d                	je     1303 <std::basic_ostream<char, std::char_traits<char> >& std::endl<char, std::char_traits<char> >(std::basic_ostream<char, std::char_traits<char> >&) [clone .isra.0]+0x43>
	  return _M_widen[static_cast<unsigned char>(__c)];
    12e6:	41 0f be 74 24 43    	movsx  esi,BYTE PTR [r12+0x43]
    12ec:	48 89 ef             	mov    rdi,rbp
    12ef:	e8 3c fd ff ff       	call   1030 <std::ostream::put(char)@plt>
    12f4:	48 83 c4 08          	add    rsp,0x8
    12f8:	5d                   	pop    rbp
    12f9:	48 89 c7             	mov    rdi,rax
    12fc:	41 5c                	pop    r12
   *  This manipulator simply calls the stream's @c flush() member function.
  */
  template<typename _CharT, typename _Traits>
    inline basic_ostream<_CharT, _Traits>&
    flush(basic_ostream<_CharT, _Traits>& __os)
    { return __os.flush(); }
    12fe:	e9 4d fd ff ff       	jmp    1050 <std::ostream::flush()@plt>
	this->_M_widen_init();
    1303:	4c 89 e7             	mov    rdi,r12
    1306:	e8 85 fd ff ff       	call   1090 <std::ctype<char>::_M_widen_init() const@plt>
	return this->do_widen(__c);
    130b:	49 8b 04 24          	mov    rax,QWORD PTR [r12]
    130f:	be 0a 00 00 00       	mov    esi,0xa
    1314:	4c 89 e7             	mov    rdi,r12
    1317:	ff 50 30             	call   QWORD PTR [rax+0x30]
    131a:	0f be f0             	movsx  esi,al
    131d:	eb cd                	jmp    12ec <std::basic_ostream<char, std::char_traits<char> >& std::endl<char, std::char_traits<char> >(std::basic_ostream<char, std::char_traits<char> >&) [clone .isra.0]+0x2c>
	__throw_bad_cast();
    131f:	e8 7c fd ff ff       	call   10a0 <std::__throw_bad_cast()@plt>
    1324:	66 2e 0f 1f 84 00 00 	nop    WORD PTR cs:[rax+rax*1+0x0]
    132b:	00 00 00 
    132e:	66 90                	xchg   ax,ax

0000000000001330 <__libc_csu_init>:
    1330:	41 57                	push   r15
    1332:	4c 8d 3d 8f 2a 00 00 	lea    r15,[rip+0x2a8f]        # 3dc8 <__frame_dummy_init_array_entry>
    1339:	41 56                	push   r14
    133b:	49 89 d6             	mov    r14,rdx
    133e:	41 55                	push   r13
    1340:	49 89 f5             	mov    r13,rsi
    1343:	41 54                	push   r12
    1345:	41 89 fc             	mov    r12d,edi
    1348:	55                   	push   rbp
    1349:	48 8d 2d 88 2a 00 00 	lea    rbp,[rip+0x2a88]        # 3dd8 <__do_global_dtors_aux_fini_array_entry>
    1350:	53                   	push   rbx
    1351:	4c 29 fd             	sub    rbp,r15
    1354:	48 83 ec 08          	sub    rsp,0x8
    1358:	e8 a3 fc ff ff       	call   1000 <_init>
    135d:	48 c1 fd 03          	sar    rbp,0x3
    1361:	74 1b                	je     137e <__libc_csu_init+0x4e>
    1363:	31 db                	xor    ebx,ebx
    1365:	0f 1f 00             	nop    DWORD PTR [rax]
    1368:	4c 89 f2             	mov    rdx,r14
    136b:	4c 89 ee             	mov    rsi,r13
    136e:	44 89 e7             	mov    edi,r12d
    1371:	41 ff 14 df          	call   QWORD PTR [r15+rbx*8]
    1375:	48 83 c3 01          	add    rbx,0x1
    1379:	48 39 dd             	cmp    rbp,rbx
    137c:	75 ea                	jne    1368 <__libc_csu_init+0x38>
    137e:	48 83 c4 08          	add    rsp,0x8
    1382:	5b                   	pop    rbx
    1383:	5d                   	pop    rbp
    1384:	41 5c                	pop    r12
    1386:	41 5d                	pop    r13
    1388:	41 5e                	pop    r14
    138a:	41 5f                	pop    r15
    138c:	c3                   	ret    
    138d:	0f 1f 00             	nop    DWORD PTR [rax]

0000000000001390 <__libc_csu_fini>:
    1390:	c3                   	ret    

Disassembly of section .fini:

0000000000001394 <_fini>:
    1394:	48 83 ec 08          	sub    rsp,0x8
    1398:	48 83 c4 08          	add    rsp,0x8
    139c:	c3                   	ret    
