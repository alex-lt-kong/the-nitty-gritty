
1_hello-world.out:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .text:

0000000000001252 <main>:
template<typename T>
T my_max(T a, T b) {
    return a > b ? a : b;
}

int main() {
    1252:	53                   	push   rbx
    1253:	48 81 ec b0 00 00 00 	sub    rsp,0xb0
    int max_int;
    double max_dbl;

    max_int = my_max(1, 3);
    125a:	be 03 00 00 00       	mov    esi,0x3
    125f:	bf 01 00 00 00       	mov    edi,0x1
    1264:	e8 9a 01 00 00       	call   1403 <int my_max<int>(int, int)>
    1269:	89 c6                	mov    esi,eax
    cout << max_int << endl;
    126b:	48 8d 3d 4e 2e 00 00 	lea    rdi,[rip+0x2e4e]        # 40c0 <std::cout@@GLIBCXX_3.4>
    1272:	e8 79 fe ff ff       	call   10f0 <std::ostream::operator<<(int)@plt>
    1277:	48 89 c7             	mov    rdi,rax
      operator<<(__ostream_type& (*__pf)(__ostream_type&))
      {
	// _GLIBCXX_RESOLVE_LIB_DEFECTS
	// DR 60. What is a formatted input function?
	// The inserters for manipulators are *not* formatted output functions.
	return __pf(*this);
    127a:	e8 d1 fd ff ff       	call   1050 <std::basic_ostream<char, std::char_traits<char> >& std::endl<char, std::char_traits<char> >(std::basic_ostream<char, std::char_traits<char> >&)@plt>

    max_dbl = my_max(1.0001, 1.0002);
    127f:	f2 0f 10 0d b9 0d 00 	movsd  xmm1,QWORD PTR [rip+0xdb9]        # 2040 <_IO_stdin_used+0x40>
    1286:	00 
    1287:	f2 0f 10 05 b9 0d 00 	movsd  xmm0,QWORD PTR [rip+0xdb9]        # 2048 <_IO_stdin_used+0x48>
    128e:	00 
    128f:	e8 78 01 00 00       	call   140c <double my_max<double>(double, double)>
       *  These functions use the stream's current locale (specifically, the
       *  @c num_get facet) to perform numeric formatting.
      */
      __ostream_type&
      operator<<(double __f)
      { return _M_insert(__f); }
    1294:	48 8d 3d 25 2e 00 00 	lea    rdi,[rip+0x2e25]        # 40c0 <std::cout@@GLIBCXX_3.4>
    129b:	e8 40 fe ff ff       	call   10e0 <std::ostream& std::ostream::_M_insert<double>(double)@plt>
    12a0:	48 89 c7             	mov    rdi,rax
	return __pf(*this);
    12a3:	e8 a8 fd ff ff       	call   1050 <std::basic_ostream<char, std::char_traits<char> >& std::endl<char, std::char_traits<char> >(std::basic_ostream<char, std::char_traits<char> >&)@plt>
    cout << max_dbl << endl;

    const char* chr_a = "Hello";
    const char* chr_b = "World!";
    cout << my_max(chr_a, chr_b) << endl;
    12a8:	48 8d 35 83 0d 00 00 	lea    rsi,[rip+0xd83]        # 2032 <_IO_stdin_used+0x32>
    12af:	48 8d 3d 83 0d 00 00 	lea    rdi,[rip+0xd83]        # 2039 <_IO_stdin_used+0x39>
    12b6:	e8 60 01 00 00       	call   141b <char const* my_max<char const*>(char const*, char const*)>
    12bb:	48 89 c6             	mov    rsi,rax
    12be:	48 8d 3d fb 2d 00 00 	lea    rdi,[rip+0x2dfb]        # 40c0 <std::cout@@GLIBCXX_3.4>
    12c5:	e8 c6 fd ff ff       	call   1090 <std::basic_ostream<char, std::char_traits<char> >& std::operator<< <std::char_traits<char> >(std::basic_ostream<char, std::char_traits<char> >&, char const*)@plt>
    12ca:	48 89 c7             	mov    rdi,rax
    12cd:	e8 7e fd ff ff       	call   1050 <std::basic_ostream<char, std::char_traits<char> >& std::endl<char, std::char_traits<char> >(std::basic_ostream<char, std::char_traits<char> >&)@plt>
    
    string a = "Hello";
    12d2:	48 8d 94 24 ae 00 00 	lea    rdx,[rsp+0xae]
    12d9:	00 
    12da:	48 8d bc 24 80 00 00 	lea    rdi,[rsp+0x80]
    12e1:	00 
    12e2:	48 8d 35 50 0d 00 00 	lea    rsi,[rip+0xd50]        # 2039 <_IO_stdin_used+0x39>
    12e9:	e8 d2 fd ff ff       	call   10c0 <std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::basic_string(char const*, std::allocator<char> const&)@plt>
    string b = "World!";
    12ee:	48 8d 94 24 af 00 00 	lea    rdx,[rsp+0xaf]
    12f5:	00 
    12f6:	48 8d 7c 24 60       	lea    rdi,[rsp+0x60]
    12fb:	48 8d 35 30 0d 00 00 	lea    rsi,[rip+0xd30]        # 2032 <_IO_stdin_used+0x32>
    1302:	e8 b9 fd ff ff       	call   10c0 <std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::basic_string(char const*, std::allocator<char> const&)@plt>
    cout << my_max(a, b) << endl;
    1307:	48 8d 74 24 60       	lea    rsi,[rsp+0x60]
    130c:	48 8d 7c 24 20       	lea    rdi,[rsp+0x20]
    1311:	e8 1a fd ff ff       	call   1030 <std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::basic_string(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&)@plt>
    1316:	48 8d b4 24 80 00 00 	lea    rsi,[rsp+0x80]
    131d:	00 
    131e:	48 89 e7             	mov    rdi,rsp
    1321:	e8 0a fd ff ff       	call   1030 <std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::basic_string(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&)@plt>
    1326:	48 8d 7c 24 40       	lea    rdi,[rsp+0x40]
    132b:	48 8d 54 24 20       	lea    rdx,[rsp+0x20]
    1330:	48 89 e6             	mov    rsi,rsp
    1333:	e8 82 01 00 00       	call   14ba <std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > my_max<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)>
    operator<<(basic_ostream<_CharT, _Traits>& __os,
	       const basic_string<_CharT, _Traits, _Alloc>& __str)
    {
      // _GLIBCXX_RESOLVE_LIB_DEFECTS
      // 586. string inserter not a formatted function
      return __ostream_insert(__os, __str.data(), __str.size());
    1338:	48 8b 54 24 48       	mov    rdx,QWORD PTR [rsp+0x48]
    133d:	48 8b 74 24 40       	mov    rsi,QWORD PTR [rsp+0x40]
    1342:	48 8d 3d 77 2d 00 00 	lea    rdi,[rip+0x2d77]        # 40c0 <std::cout@@GLIBCXX_3.4>
    1349:	e8 52 fd ff ff       	call   10a0 <std::basic_ostream<char, std::char_traits<char> >& std::__ostream_insert<char, std::char_traits<char> >(std::basic_ostream<char, std::char_traits<char> >&, char const*, long)@plt>
    134e:	48 89 c7             	mov    rdi,rax
    1351:	e8 fa fc ff ff       	call   1050 <std::basic_ostream<char, std::char_traits<char> >& std::endl<char, std::char_traits<char> >(std::basic_ostream<char, std::char_traits<char> >&)@plt>
    1356:	eb 52                	jmp    13aa <main+0x158>
      { _M_dispose(); }
    1358:	48 89 c3             	mov    rbx,rax
    135b:	48 8d 7c 24 40       	lea    rdi,[rsp+0x40]
    1360:	e8 4b fd ff ff       	call   10b0 <std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_dispose()@plt>
    1365:	48 89 e7             	mov    rdi,rsp
    1368:	e8 43 fd ff ff       	call   10b0 <std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_dispose()@plt>
    136d:	48 8d 7c 24 20       	lea    rdi,[rsp+0x20]
    1372:	e8 39 fd ff ff       	call   10b0 <std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_dispose()@plt>
    1377:	48 8d 7c 24 60       	lea    rdi,[rsp+0x60]
    137c:	e8 2f fd ff ff       	call   10b0 <std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_dispose()@plt>
    1381:	48 8d bc 24 80 00 00 	lea    rdi,[rsp+0x80]
    1388:	00 
    1389:	e8 22 fd ff ff       	call   10b0 <std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_dispose()@plt>
      template<typename _Tp1>
	_GLIBCXX20_CONSTEXPR
	new_allocator(const new_allocator<_Tp1>&) _GLIBCXX_USE_NOEXCEPT { }

#if __cplusplus <= 201703L
      ~new_allocator() _GLIBCXX_USE_NOEXCEPT { }
    138e:	48 89 df             	mov    rdi,rbx
    1391:	e8 6a fd ff ff       	call   1100 <_Unwind_Resume@plt>
    1396:	48 89 c3             	mov    rbx,rax
    1399:	eb ca                	jmp    1365 <main+0x113>
    139b:	48 89 c3             	mov    rbx,rax
    139e:	eb cd                	jmp    136d <main+0x11b>
    13a0:	48 89 c3             	mov    rbx,rax
    13a3:	eb d2                	jmp    1377 <main+0x125>
    13a5:	48 89 c3             	mov    rbx,rax
    13a8:	eb d7                	jmp    1381 <main+0x12f>
    13aa:	48 8d 7c 24 40       	lea    rdi,[rsp+0x40]
    13af:	e8 fc fc ff ff       	call   10b0 <std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_dispose()@plt>
    13b4:	48 89 e7             	mov    rdi,rsp
    13b7:	e8 f4 fc ff ff       	call   10b0 <std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_dispose()@plt>
    13bc:	48 8d 7c 24 20       	lea    rdi,[rsp+0x20]
    13c1:	e8 ea fc ff ff       	call   10b0 <std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_dispose()@plt>
    13c6:	48 8d 7c 24 60       	lea    rdi,[rsp+0x60]
    13cb:	e8 e0 fc ff ff       	call   10b0 <std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_dispose()@plt>
    13d0:	48 8d bc 24 80 00 00 	lea    rdi,[rsp+0x80]
    13d7:	00 
    13d8:	e8 d3 fc ff ff       	call   10b0 <std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_dispose()@plt>
    return 0;
    13dd:	b8 00 00 00 00       	mov    eax,0x0
    13e2:	48 81 c4 b0 00 00 00 	add    rsp,0xb0
    13e9:	5b                   	pop    rbx
    13ea:	c3                   	ret    

Disassembly of section .fini:
