
1_hello-world.out:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .text:

00000000000011f0 <main>:
template<typename T>
T my_max(T a, T b) {
    return a > b ? a : b;
}

int main() {
    11f0:	41 57                	push   r15
    int max_int;
    double max_dbl;

    max_int = my_max(1, 3);
    cout << max_int << endl;
    11f2:	be 03 00 00 00       	mov    esi,0x3
    11f7:	48 8d 3d c2 2e 00 00 	lea    rdi,[rip+0x2ec2]        # 40c0 <std::cout@@GLIBCXX_3.4>
int main() {
    11fe:	41 56                	push   r14
    1200:	41 55                	push   r13
    1202:	41 54                	push   r12
    max_dbl = my_max(1.0001, 1.0002);
    cout << max_dbl << endl;

    const char* chr_a = "Hello";
    const char* chr_b = "World!";
    cout << my_max(chr_a, chr_b) << endl;
    1204:	4c 8d 25 2d 0e 00 00 	lea    r12,[rip+0xe2d]        # 2038 <_IO_stdin_used+0x38>
int main() {
    120b:	55                   	push   rbp
    cout << my_max(chr_a, chr_b) << endl;
    120c:	48 8d 2d 1f 0e 00 00 	lea    rbp,[rip+0xe1f]        # 2032 <_IO_stdin_used+0x32>
int main() {
    1213:	53                   	push   rbx
    1214:	48 81 ec c8 00 00 00 	sub    rsp,0xc8
    cout << max_int << endl;
    121b:	e8 f0 fe ff ff       	call   1110 <std::ostream::operator<<(int)@plt>
    1220:	48 89 c7             	mov    rdi,rax
      operator<<(__ostream_type& (*__pf)(__ostream_type&))
      {
	// _GLIBCXX_RESOLVE_LIB_DEFECTS
	// DR 60. What is a formatted input function?
	// The inserters for manipulators are *not* formatted output functions.
	return __pf(*this);
    1223:	e8 e8 04 00 00       	call   1710 <std::basic_ostream<char, std::char_traits<char> >& std::endl<char, std::char_traits<char> >(std::basic_ostream<char, std::char_traits<char> >&) [clone .isra.0]>
       *  These functions use the stream's current locale (specifically, the
       *  @c num_get facet) to perform numeric formatting.
      */
      __ostream_type&
      operator<<(double __f)
      { return _M_insert(__f); }
    1228:	f2 0f 10 05 10 0e 00 	movsd  xmm0,QWORD PTR [rip+0xe10]        # 2040 <_IO_stdin_used+0x40>
    122f:	00 
    1230:	48 8d 3d 89 2e 00 00 	lea    rdi,[rip+0x2e89]        # 40c0 <std::cout@@GLIBCXX_3.4>
    1237:	e8 c4 fe ff ff       	call   1100 <std::ostream& std::ostream::_M_insert<double>(double)@plt>
    123c:	48 89 c7             	mov    rdi,rax
	return __pf(*this);
    123f:	e8 cc 04 00 00       	call   1710 <std::basic_ostream<char, std::char_traits<char> >& std::endl<char, std::char_traits<char> >(std::basic_ostream<char, std::char_traits<char> >&) [clone .isra.0]>
    cout << my_max(chr_a, chr_b) << endl;
    1244:	4c 39 e5             	cmp    rbp,r12
    1247:	4c 89 e6             	mov    rsi,r12
    124a:	48 8d 3d 6f 2e 00 00 	lea    rdi,[rip+0x2e6f]        # 40c0 <std::cout@@GLIBCXX_3.4>
    1251:	48 0f 43 f5          	cmovae rsi,rbp
    1255:	e8 46 fe ff ff       	call   10a0 <std::basic_ostream<char, std::char_traits<char> >& std::operator<< <std::char_traits<char> >(std::basic_ostream<char, std::char_traits<char> >&, char const*)@plt>
    125a:	48 89 c7             	mov    rdi,rax
    125d:	e8 ae 04 00 00       	call   1710 <std::basic_ostream<char, std::char_traits<char> >& std::endl<char, std::char_traits<char> >(std::basic_ostream<char, std::char_traits<char> >&) [clone .isra.0]>
    
    string a = "Hello";
    1262:	48 8d 7c 24 20       	lea    rdi,[rsp+0x20]
    1267:	48 89 ee             	mov    rsi,rbp
    126a:	e8 01 04 00 00       	call   1670 <std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::basic_string(char const*, std::allocator<char> const&) [clone .isra.0]>
    string b = "World!";
    126f:	48 8d 7c 24 40       	lea    rdi,[rsp+0x40]
    1274:	4c 89 e6             	mov    rsi,r12
    1277:	e8 f4 03 00 00       	call   1670 <std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::basic_string(char const*, std::allocator<char> const&) [clone .isra.0]>
      _M_length(size_type __length)
      { _M_string_length = __length; }

      pointer
      _M_data() const
      { return _M_dataplus._M_p; }
    127c:	48 8b 74 24 40       	mov    rsi,QWORD PTR [rsp+0x40]
       *  @param  __str  Source string.
       */
      basic_string(const basic_string& __str)
      : _M_dataplus(_M_local_data(),
		    _Alloc_traits::_S_select_on_copy(__str._M_get_allocator()))
      { _M_construct(__str._M_data(), __str._M_data() + __str.length()); }
    1281:	48 8b 54 24 48       	mov    rdx,QWORD PTR [rsp+0x48]
	: allocator_type(std::move(__a)), _M_p(__dat) { }
    1286:	48 8d 9c 24 80 00 00 	lea    rbx,[rsp+0x80]
    128d:	00 
    128e:	4c 8d ac 24 90 00 00 	lea    r13,[rsp+0x90]
    1295:	00 
          _M_construct(__beg, __end, _Tag());
    1296:	48 89 df             	mov    rdi,rbx
	: allocator_type(std::move(__a)), _M_p(__dat) { }
    1299:	4c 89 ac 24 80 00 00 	mov    QWORD PTR [rsp+0x80],r13
    12a0:	00 
      { _M_construct(__str._M_data(), __str._M_data() + __str.length()); }
    12a1:	48 01 f2             	add    rdx,rsi
          _M_construct(__beg, __end, _Tag());
    12a4:	e8 d7 04 00 00       	call   1780 <void std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_construct<char*>(char*, char*, std::forward_iterator_tag) [clone .isra.0]>
      { return _M_dataplus._M_p; }
    12a9:	48 8b 74 24 20       	mov    rsi,QWORD PTR [rsp+0x20]
      { _M_construct(__str._M_data(), __str._M_data() + __str.length()); }
    12ae:	48 8b 54 24 28       	mov    rdx,QWORD PTR [rsp+0x28]
	: allocator_type(std::move(__a)), _M_p(__dat) { }
    12b3:	4c 8d 74 24 60       	lea    r14,[rsp+0x60]
    12b8:	48 8d 6c 24 70       	lea    rbp,[rsp+0x70]
          _M_construct(__beg, __end, _Tag());
    12bd:	4c 89 f7             	mov    rdi,r14
	: allocator_type(std::move(__a)), _M_p(__dat) { }
    12c0:	48 89 6c 24 60       	mov    QWORD PTR [rsp+0x60],rbp
      { _M_construct(__str._M_data(), __str._M_data() + __str.length()); }
    12c5:	48 01 f2             	add    rdx,rsi
          _M_construct(__beg, __end, _Tag());
    12c8:	e8 b3 04 00 00       	call   1780 <void std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_construct<char*>(char*, char*, std::forward_iterator_tag) [clone .isra.0]>
   *  @param __rhs  Second string.
   *  @return  True if @a __lhs follows @a __rhs.  False otherwise.
   */
  template<typename _CharT, typename _Traits, typename _Alloc>
    inline bool
    operator>(const basic_string<_CharT, _Traits, _Alloc>& __lhs,
    12cd:	4c 8b 64 24 68       	mov    r12,QWORD PTR [rsp+0x68]
    12d2:	48 8b 8c 24 88 00 00 	mov    rcx,QWORD PTR [rsp+0x88]
    12d9:	00 
      compare(const basic_string& __str) const
    12da:	49 39 cc             	cmp    r12,rcx
    12dd:	48 89 ca             	mov    rdx,rcx
    12e0:	49 0f 46 d4          	cmovbe rdx,r12
      }

      static _GLIBCXX17_CONSTEXPR int
      compare(const char_type* __s1, const char_type* __s2, size_t __n)
      {
	if (__n == 0)
    12e4:	48 85 d2             	test   rdx,rdx
    12e7:	0f 84 95 01 00 00    	je     1482 <main+0x292>
    12ed:	4c 8b 7c 24 60       	mov    r15,QWORD PTR [rsp+0x60]
	if (__builtin_constant_p(__n)
	    && __constant_char_array_p(__s1, __n)
	    && __constant_char_array_p(__s2, __n))
	  return __gnu_cxx::char_traits<char_type>::compare(__s1, __s2, __n);
#endif
	return __builtin_memcmp(__s1, __s2, __n);
    12f2:	48 8b b4 24 80 00 00 	mov    rsi,QWORD PTR [rsp+0x80]
    12f9:	00 
    12fa:	48 89 4c 24 08       	mov    QWORD PTR [rsp+0x8],rcx
    12ff:	4c 89 ff             	mov    rdi,r15
    1302:	48 89 34 24          	mov    QWORD PTR [rsp],rsi
    1306:	e8 45 fd ff ff       	call   1050 <memcmp@plt>
	if (!__r)
    130b:	48 8b 34 24          	mov    rsi,QWORD PTR [rsp]
    130f:	48 8b 4c 24 08       	mov    rcx,QWORD PTR [rsp+0x8]
    1314:	85 c0                	test   eax,eax
    1316:	0f 85 4f 01 00 00    	jne    146b <main+0x27b>
	const difference_type __d = difference_type(__n1 - __n2);
    131c:	4c 89 e0             	mov    rax,r12
    131f:	48 29 c8             	sub    rax,rcx
	if (__d > __gnu_cxx::__numeric_traits<int>::__max)
    1322:	48 3d ff ff ff 7f    	cmp    rax,0x7fffffff
    1328:	0f 8f 3f 01 00 00    	jg     146d <main+0x27d>
	else if (__d < __gnu_cxx::__numeric_traits<int>::__min)
    132e:	48 3d 00 00 00 80    	cmp    rax,0xffffffff80000000
    1334:	0f 8c b1 01 00 00    	jl     14eb <main+0x2fb>
    return a > b ? a : b;
    133a:	85 c0                	test   eax,eax
    133c:	49 0f 4f de          	cmovg  rbx,r14
      { return _M_dataplus._M_p; }
    1340:	4c 8b 3b             	mov    r15,QWORD PTR [rbx]
      { return _M_string_length; }
    1343:	4c 8b 63 08          	mov    r12,QWORD PTR [rbx+0x8]
	: allocator_type(std::move(__a)), _M_p(__dat) { }
    1347:	4c 8d b4 24 b0 00 00 	lea    r14,[rsp+0xb0]
    134e:	00 
    134f:	4c 89 b4 24 a0 00 00 	mov    QWORD PTR [rsp+0xa0],r14
    1356:	00 
      basic_string<_CharT, _Traits, _Alloc>::
      _M_construct(_InIterator __beg, _InIterator __end,
		   std::forward_iterator_tag)
      {
	// NB: Not required, but considered best practice.
	if (__gnu_cxx::__is_null_pointer(__beg) && __beg != __end)
    1357:	4c 89 f8             	mov    rax,r15
    135a:	4c 01 e0             	add    rax,r12
    135d:	74 09                	je     1368 <main+0x178>
    135f:	4d 85 ff             	test   r15,r15
    1362:	0f 84 a6 01 00 00    	je     150e <main+0x31e>
	  std::__throw_logic_error(__N("basic_string::"
				       "_M_construct null not valid"));

	size_type __dnew = static_cast<size_type>(std::distance(__beg, __end));
    1368:	4c 89 64 24 18       	mov    QWORD PTR [rsp+0x18],r12

	if (__dnew > size_type(_S_local_capacity))
    136d:	49 83 fc 0f          	cmp    r12,0xf
    1371:	0f 87 38 01 00 00    	ja     14af <main+0x2bf>
	if (__n == 1)
    1377:	49 83 fc 01          	cmp    r12,0x1
    137b:	0f 85 20 01 00 00    	jne    14a1 <main+0x2b1>
      { __c1 = __c2; }
    1381:	41 0f b6 07          	movzx  eax,BYTE PTR [r15]
    1385:	88 84 24 b0 00 00 00 	mov    BYTE PTR [rsp+0xb0],al
	  {
	    _M_dispose();
	    __throw_exception_again;
	  }

	_M_set_length(__dnew);
    138c:	48 8b 44 24 18       	mov    rax,QWORD PTR [rsp+0x18]
    1391:	48 8b 94 24 a0 00 00 	mov    rdx,QWORD PTR [rsp+0xa0]
    1398:	00 
    operator<<(basic_ostream<_CharT, _Traits>& __os,
	       const basic_string<_CharT, _Traits, _Alloc>& __str)
    {
      // _GLIBCXX_RESOLVE_LIB_DEFECTS
      // 586. string inserter not a formatted function
      return __ostream_insert(__os, __str.data(), __str.size());
    1399:	48 8d 3d 20 2d 00 00 	lea    rdi,[rip+0x2d20]        # 40c0 <std::cout@@GLIBCXX_3.4>
      { _M_string_length = __length; }
    13a0:	48 89 84 24 a8 00 00 	mov    QWORD PTR [rsp+0xa8],rax
    13a7:	00 
    13a8:	c6 04 02 00          	mov    BYTE PTR [rdx+rax*1],0x0
      return __ostream_insert(__os, __str.data(), __str.size());
    13ac:	48 8b 94 24 a8 00 00 	mov    rdx,QWORD PTR [rsp+0xa8]
    13b3:	00 
    13b4:	48 8b b4 24 a0 00 00 	mov    rsi,QWORD PTR [rsp+0xa0]
    13bb:	00 
    13bc:	e8 ff fc ff ff       	call   10c0 <std::basic_ostream<char, std::char_traits<char> >& std::__ostream_insert<char, std::char_traits<char> >(std::basic_ostream<char, std::char_traits<char> >&, char const*, long)@plt>
    13c1:	48 89 c7             	mov    rdi,rax
    13c4:	e8 47 03 00 00       	call   1710 <std::basic_ostream<char, std::char_traits<char> >& std::endl<char, std::char_traits<char> >(std::basic_ostream<char, std::char_traits<char> >&) [clone .isra.0]>
      { return _M_dataplus._M_p; }
    13c9:	48 8b bc 24 a0 00 00 	mov    rdi,QWORD PTR [rsp+0xa0]
    13d0:	00 
	if (!_M_is_local())
    13d1:	4c 39 f7             	cmp    rdi,r14
    13d4:	74 11                	je     13e7 <main+0x1f7>
      { _Alloc_traits::deallocate(_M_get_allocator(), _M_data(), __size + 1); }
    13d6:	48 8b 84 24 b0 00 00 	mov    rax,QWORD PTR [rsp+0xb0]
    13dd:	00 
    13de:	48 8d 70 01          	lea    rsi,[rax+0x1]
# endif
			      std::align_val_t(alignof(_Tp)));
	    return;
	  }
#endif
	::operator delete(__p
    13e2:	e8 c9 fc ff ff       	call   10b0 <operator delete(void*, unsigned long)@plt>
      { return _M_dataplus._M_p; }
    13e7:	48 8b 7c 24 60       	mov    rdi,QWORD PTR [rsp+0x60]
	if (!_M_is_local())
    13ec:	48 39 ef             	cmp    rdi,rbp
    13ef:	74 0e                	je     13ff <main+0x20f>
      { _Alloc_traits::deallocate(_M_get_allocator(), _M_data(), __size + 1); }
    13f1:	48 8b 44 24 70       	mov    rax,QWORD PTR [rsp+0x70]
    13f6:	48 8d 70 01          	lea    rsi,[rax+0x1]
    13fa:	e8 b1 fc ff ff       	call   10b0 <operator delete(void*, unsigned long)@plt>
      { return _M_dataplus._M_p; }
    13ff:	48 8b bc 24 80 00 00 	mov    rdi,QWORD PTR [rsp+0x80]
    1406:	00 
	if (!_M_is_local())
    1407:	4c 39 ef             	cmp    rdi,r13
    140a:	74 11                	je     141d <main+0x22d>
      { _Alloc_traits::deallocate(_M_get_allocator(), _M_data(), __size + 1); }
    140c:	48 8b 84 24 90 00 00 	mov    rax,QWORD PTR [rsp+0x90]
    1413:	00 
    1414:	48 8d 70 01          	lea    rsi,[rax+0x1]
    1418:	e8 93 fc ff ff       	call   10b0 <operator delete(void*, unsigned long)@plt>
      { return _M_dataplus._M_p; }
    141d:	48 8b 7c 24 40       	mov    rdi,QWORD PTR [rsp+0x40]
	if (!_M_is_local())
    1422:	48 8d 44 24 50       	lea    rax,[rsp+0x50]
    1427:	48 39 c7             	cmp    rdi,rax
    142a:	74 0e                	je     143a <main+0x24a>
      { _Alloc_traits::deallocate(_M_get_allocator(), _M_data(), __size + 1); }
    142c:	48 8b 44 24 50       	mov    rax,QWORD PTR [rsp+0x50]
    1431:	48 8d 70 01          	lea    rsi,[rax+0x1]
    1435:	e8 76 fc ff ff       	call   10b0 <operator delete(void*, unsigned long)@plt>
      { return _M_dataplus._M_p; }
    143a:	48 8b 7c 24 20       	mov    rdi,QWORD PTR [rsp+0x20]
	if (!_M_is_local())
    143f:	48 8d 44 24 30       	lea    rax,[rsp+0x30]
    1444:	48 39 c7             	cmp    rdi,rax
    1447:	74 0e                	je     1457 <main+0x267>
      { _Alloc_traits::deallocate(_M_get_allocator(), _M_data(), __size + 1); }
    1449:	48 8b 44 24 30       	mov    rax,QWORD PTR [rsp+0x30]
    144e:	48 8d 70 01          	lea    rsi,[rax+0x1]
    1452:	e8 59 fc ff ff       	call   10b0 <operator delete(void*, unsigned long)@plt>
    cout << my_max(a, b) << endl;
    return 0;
    1457:	48 81 c4 c8 00 00 00 	add    rsp,0xc8
    145e:	31 c0                	xor    eax,eax
    1460:	5b                   	pop    rbx
    1461:	5d                   	pop    rbp
    1462:	41 5c                	pop    r12
    1464:	41 5d                	pop    r13
    1466:	41 5e                	pop    r14
    1468:	41 5f                	pop    r15
    146a:	c3                   	ret    
    return a > b ? a : b;
    146b:	7e 7e                	jle    14eb <main+0x2fb>
	: allocator_type(std::move(__a)), _M_p(__dat) { }
    146d:	4c 8d b4 24 b0 00 00 	lea    r14,[rsp+0xb0]
    1474:	00 
    1475:	4c 89 b4 24 a0 00 00 	mov    QWORD PTR [rsp+0xa0],r14
    147c:	00 
      basic_string<_CharT, _Traits, _Alloc>::
    147d:	e9 e6 fe ff ff       	jmp    1368 <main+0x178>
	const difference_type __d = difference_type(__n1 - __n2);
    1482:	4c 89 e0             	mov    rax,r12
    1485:	48 29 c8             	sub    rax,rcx
	if (__d > __gnu_cxx::__numeric_traits<int>::__max)
    1488:	48 3d ff ff ff 7f    	cmp    rax,0x7fffffff
    148e:	7f 76                	jg     1506 <main+0x316>
	else if (__d < __gnu_cxx::__numeric_traits<int>::__min)
    1490:	48 3d 00 00 00 80    	cmp    rax,0xffffffff80000000
    1496:	0f 8d 9e fe ff ff    	jge    133a <main+0x14a>
    149c:	e9 9f fe ff ff       	jmp    1340 <main+0x150>
      }

      static _GLIBCXX20_CONSTEXPR char_type*
      copy(char_type* __s1, const char_type* __s2, size_t __n)
      {
	if (__n == 0)
    14a1:	4d 85 e4             	test   r12,r12
    14a4:	0f 84 e2 fe ff ff    	je     138c <main+0x19c>
      { return _M_dataplus._M_p; }
    14aa:	4c 89 f7             	mov    rdi,r14
    14ad:	eb 2c                	jmp    14db <main+0x2eb>
	    _M_data(_M_create(__dnew, size_type(0)));
    14af:	48 8d 74 24 18       	lea    rsi,[rsp+0x18]
    14b4:	48 8d bc 24 a0 00 00 	lea    rdi,[rsp+0xa0]
    14bb:	00 
    14bc:	31 d2                	xor    edx,edx
    14be:	e8 6d fc ff ff       	call   1130 <std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_create(unsigned long&, unsigned long)@plt>
      { _M_dataplus._M_p = __p; }
    14c3:	48 89 84 24 a0 00 00 	mov    QWORD PTR [rsp+0xa0],rax
    14ca:	00 
    14cb:	48 89 c7             	mov    rdi,rax
      { _M_allocated_capacity = __capacity; }
    14ce:	48 8b 44 24 18       	mov    rax,QWORD PTR [rsp+0x18]
    14d3:	48 89 84 24 b0 00 00 	mov    QWORD PTR [rsp+0xb0],rax
    14da:	00 
	  return __s1;
#ifdef __cpp_lib_is_constant_evaluated
	if (std::is_constant_evaluated())
	  return __gnu_cxx::char_traits<char_type>::copy(__s1, __s2, __n);
#endif
	return static_cast<char_type*>(__builtin_memcpy(__s1, __s2, __n));
    14db:	4c 89 e2             	mov    rdx,r12
    14de:	4c 89 fe             	mov    rsi,r15
    14e1:	e8 9a fb ff ff       	call   1080 <memcpy@plt>
    14e6:	e9 a1 fe ff ff       	jmp    138c <main+0x19c>
	: allocator_type(std::move(__a)), _M_p(__dat) { }
    14eb:	4c 8d b4 24 b0 00 00 	lea    r14,[rsp+0xb0]
    14f2:	00 
      { return _M_dataplus._M_p; }
    14f3:	49 89 f7             	mov    r15,rsi
      { return _M_string_length; }
    14f6:	49 89 cc             	mov    r12,rcx
	: allocator_type(std::move(__a)), _M_p(__dat) { }
    14f9:	4c 89 b4 24 a0 00 00 	mov    QWORD PTR [rsp+0xa0],r14
    1500:	00 
      basic_string<_CharT, _Traits, _Alloc>::
    1501:	e9 62 fe ff ff       	jmp    1368 <main+0x178>
    1506:	4c 89 f3             	mov    rbx,r14
    1509:	e9 32 fe ff ff       	jmp    1340 <main+0x150>
	  std::__throw_logic_error(__N("basic_string::"
    150e:	48 8d 3d f3 0a 00 00 	lea    rdi,[rip+0xaf3]        # 2008 <_IO_stdin_used+0x8>
    1515:	e8 56 fb ff ff       	call   1070 <std::__throw_logic_error(char const*)@plt>
      { return _M_dataplus._M_p; }
    151a:	48 89 c3             	mov    rbx,rax
    151d:	e9 2e fc ff ff       	jmp    1150 <main.cold>
    1522:	48 89 c3             	mov    rbx,rax
    1525:	e9 44 fc ff ff       	jmp    116e <main.cold+0x1e>
    152a:	48 89 c5             	mov    rbp,rax
    152d:	e9 75 fc ff ff       	jmp    11a7 <main.cold+0x57>
    1532:	48 89 c5             	mov    rbp,rax
    1535:	e9 4f fc ff ff       	jmp    1189 <main.cold+0x39>
    153a:	48 89 c5             	mov    rbp,rax
    153d:	e9 82 fc ff ff       	jmp    11c4 <main.cold+0x74>

Disassembly of section .fini:
