
main.out:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .text:

00000000000011c5 <main>:
#include <string>
#include <iostream>

using namespace std;

int main() {
    11c5:	53                   	push   rbx
    11c6:	48 83 c4 80          	add    rsp,0xffffffffffffff80
    string s0 = to_string(12);
    11ca:	48 8d 7c 24 60       	lea    rdi,[rsp+0x60]
    11cf:	be 0c 00 00 00       	mov    esi,0xc
    11d4:	e8 20 02 00 00       	call   13f9 <std::__cxx11::to_string(int)>
    cout << "s0: " << s0 << endl;
    11d9:	48 8d 35 24 0e 00 00 	lea    rsi,[rip+0xe24]        # 2004 <_IO_stdin_used+0x4>
    11e0:	48 8d 3d 99 2e 00 00 	lea    rdi,[rip+0x2e99]        # 4080 <std::cout@@GLIBCXX_3.4>
    11e7:	e8 74 fe ff ff       	call   1060 <std::basic_ostream<char, std::char_traits<char> >& std::operator<< <std::char_traits<char> >(std::basic_ostream<char, std::char_traits<char> >&, char const*)@plt>
    11ec:	48 89 c7             	mov    rdi,rax
    operator<<(basic_ostream<_CharT, _Traits>& __os,
	       const basic_string<_CharT, _Traits, _Alloc>& __str)
    {
      // _GLIBCXX_RESOLVE_LIB_DEFECTS
      // 586. string inserter not a formatted function
      return __ostream_insert(__os, __str.data(), __str.size());
    11ef:	48 8b 54 24 68       	mov    rdx,QWORD PTR [rsp+0x68]
    11f4:	48 8b 74 24 60       	mov    rsi,QWORD PTR [rsp+0x60]
    11f9:	e8 72 fe ff ff       	call   1070 <std::basic_ostream<char, std::char_traits<char> >& std::__ostream_insert<char, std::char_traits<char> >(std::basic_ostream<char, std::char_traits<char> >&, char const*, long)@plt>
    11fe:	48 89 c7             	mov    rdi,rax
      operator<<(__ostream_type& (*__pf)(__ostream_type&))
      {
	// _GLIBCXX_RESOLVE_LIB_DEFECTS
	// DR 60. What is a formatted input function?
	// The inserters for manipulators are *not* formatted output functions.
	return __pf(*this);
    1201:	e8 3a fe ff ff       	call   1040 <std::basic_ostream<char, std::char_traits<char> >& std::endl<char, std::char_traits<char> >(std::basic_ostream<char, std::char_traits<char> >&)@plt>
    string s1 = to_string(34) + to_string(56);
    1206:	48 8d 7c 24 20       	lea    rdi,[rsp+0x20]
    120b:	be 38 00 00 00       	mov    esi,0x38
    1210:	e8 e4 01 00 00       	call   13f9 <std::__cxx11::to_string(int)>
    1215:	48 89 e7             	mov    rdi,rsp
    1218:	be 22 00 00 00       	mov    esi,0x22
    121d:	e8 d7 01 00 00       	call   13f9 <std::__cxx11::to_string(int)>
      { return _M_string_length; }
    1222:	4c 8b 44 24 08       	mov    r8,QWORD PTR [rsp+0x8]
    1227:	48 8b 54 24 28       	mov    rdx,QWORD PTR [rsp+0x28]
	  const auto __size = __lhs.size() + __rhs.size();
    122c:	49 8d 0c 10          	lea    rcx,[r8+rdx*1]
      { return _M_dataplus._M_p; }
    1230:	48 8b 34 24          	mov    rsi,QWORD PTR [rsp]
	return _M_is_local() ? size_type(_S_local_capacity)
    1234:	48 8d 44 24 10       	lea    rax,[rsp+0x10]
    1239:	48 39 c6             	cmp    rsi,rax
    123c:	b8 0f 00 00 00       	mov    eax,0xf
    1241:	48 0f 45 44 24 10    	cmovne rax,QWORD PTR [rsp+0x10]
	  if (__size > __lhs.capacity() && __size <= __rhs.capacity())
    1247:	48 39 c1             	cmp    rcx,rax
    124a:	76 16                	jbe    1262 <main+0x9d>
	return _M_is_local() ? size_type(_S_local_capacity)
    124c:	48 8d 44 24 30       	lea    rax,[rsp+0x30]
    1251:	48 39 44 24 20       	cmp    QWORD PTR [rsp+0x20],rax
    1256:	74 19                	je     1271 <main+0xac>
    1258:	48 8b 44 24 30       	mov    rax,QWORD PTR [rsp+0x30]
	  if (__size > __lhs.capacity() && __size <= __rhs.capacity())
    125d:	48 39 c1             	cmp    rcx,rax
    1260:	76 16                	jbe    1278 <main+0xb3>
      { return _M_append(__str._M_data(), __str.size()); }
    1262:	48 89 e7             	mov    rdi,rsp
    1265:	48 8b 74 24 20       	mov    rsi,QWORD PTR [rsp+0x20]
    126a:	e8 c1 fd ff ff       	call   1030 <std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_append(char const*, unsigned long)@plt>
    126f:	eb 60                	jmp    12d1 <main+0x10c>
	return _M_is_local() ? size_type(_S_local_capacity)
    1271:	b8 0f 00 00 00       	mov    eax,0xf
    1276:	eb e5                	jmp    125d <main+0x98>
	return _M_replace(_M_check(__pos, "basic_string::replace"),
    1278:	48 8d 7c 24 20       	lea    rdi,[rsp+0x20]
    127d:	48 89 f1             	mov    rcx,rsi
    1280:	ba 00 00 00 00       	mov    edx,0x0
    1285:	be 00 00 00 00       	mov    esi,0x0
    128a:	e8 21 fe ff ff       	call   10b0 <std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_replace(unsigned long, unsigned long, char const*, unsigned long)@plt>
	: allocator_type(std::move(__a)), _M_p(__dat) { }
    128f:	48 8d 54 24 50       	lea    rdx,[rsp+0x50]
    1294:	48 89 54 24 40       	mov    QWORD PTR [rsp+0x40],rdx
      { return _M_dataplus._M_p; }
    1299:	48 8b 08             	mov    rcx,QWORD PTR [rax]
	return std::pointer_traits<const_pointer>::pointer_to(*_M_local_buf);
    129c:	48 8d 50 10          	lea    rdx,[rax+0x10]
	if (__str._M_is_local())
    12a0:	48 39 d1             	cmp    rcx,rdx
    12a3:	0f 84 8b 00 00 00    	je     1334 <main+0x16f>
      { _M_dataplus._M_p = __p; }
    12a9:	48 89 4c 24 40       	mov    QWORD PTR [rsp+0x40],rcx
	    _M_capacity(__str._M_allocated_capacity);
    12ae:	48 8b 48 10          	mov    rcx,QWORD PTR [rax+0x10]
      { _M_allocated_capacity = __capacity; }
    12b2:	48 89 4c 24 50       	mov    QWORD PTR [rsp+0x50],rcx
      { return _M_string_length; }
    12b7:	48 8b 48 08          	mov    rcx,QWORD PTR [rax+0x8]
      { _M_string_length = __length; }
    12bb:	48 89 4c 24 48       	mov    QWORD PTR [rsp+0x48],rcx
      { _M_dataplus._M_p = __p; }
    12c0:	48 89 10             	mov    QWORD PTR [rax],rdx
      { _M_string_length = __length; }
    12c3:	48 c7 40 08 00 00 00 	mov    QWORD PTR [rax+0x8],0x0
    12ca:	00 
      using comparison_category = strong_ordering;
#endif

      static _GLIBCXX17_CONSTEXPR void
      assign(char_type& __c1, const char_type& __c2) _GLIBCXX_NOEXCEPT
      { __c1 = __c2; }
    12cb:	c6 40 10 00          	mov    BYTE PTR [rax+0x10],0x0
      }
    12cf:	eb 3c                	jmp    130d <main+0x148>
	: allocator_type(std::move(__a)), _M_p(__dat) { }
    12d1:	48 8d 54 24 50       	lea    rdx,[rsp+0x50]
    12d6:	48 89 54 24 40       	mov    QWORD PTR [rsp+0x40],rdx
      { return _M_dataplus._M_p; }
    12db:	48 8b 08             	mov    rcx,QWORD PTR [rax]
	return std::pointer_traits<const_pointer>::pointer_to(*_M_local_buf);
    12de:	48 8d 50 10          	lea    rdx,[rax+0x10]
	if (__str._M_is_local())
    12e2:	48 39 d1             	cmp    rcx,rdx
    12e5:	74 5c                	je     1343 <main+0x17e>
      { _M_dataplus._M_p = __p; }
    12e7:	48 89 4c 24 40       	mov    QWORD PTR [rsp+0x40],rcx
	    _M_capacity(__str._M_allocated_capacity);
    12ec:	48 8b 48 10          	mov    rcx,QWORD PTR [rax+0x10]
      { _M_allocated_capacity = __capacity; }
    12f0:	48 89 4c 24 50       	mov    QWORD PTR [rsp+0x50],rcx
      { return _M_string_length; }
    12f5:	48 8b 48 08          	mov    rcx,QWORD PTR [rax+0x8]
      { _M_string_length = __length; }
    12f9:	48 89 4c 24 48       	mov    QWORD PTR [rsp+0x48],rcx
      { _M_dataplus._M_p = __p; }
    12fe:	48 89 10             	mov    QWORD PTR [rax],rdx
      { _M_string_length = __length; }
    1301:	48 c7 40 08 00 00 00 	mov    QWORD PTR [rax+0x8],0x0
    1308:	00 
    1309:	c6 40 10 00          	mov    BYTE PTR [rax+0x10],0x0
      { _M_dispose(); }
    130d:	48 89 e7             	mov    rdi,rsp
    1310:	e8 7b fd ff ff       	call   1090 <std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_dispose()@plt>
    1315:	48 8d 7c 24 20       	lea    rdi,[rsp+0x20]
    131a:	e8 71 fd ff ff       	call   1090 <std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_dispose()@plt>
    cout << "s1: " << s1 << endl;
    131f:	48 8d 35 e3 0c 00 00 	lea    rsi,[rip+0xce3]        # 2009 <_IO_stdin_used+0x9>
    1326:	48 8d 3d 53 2d 00 00 	lea    rdi,[rip+0x2d53]        # 4080 <std::cout@@GLIBCXX_3.4>
    132d:	e8 2e fd ff ff       	call   1060 <std::basic_ostream<char, std::char_traits<char> >& std::operator<< <std::char_traits<char> >(std::basic_ostream<char, std::char_traits<char> >&, char const*)@plt>
    1332:	eb 1b                	jmp    134f <main+0x18a>
	  return __s1;
#ifdef __cpp_lib_is_constant_evaluated
	if (std::is_constant_evaluated())
	  return __gnu_cxx::char_traits<char_type>::copy(__s1, __s2, __n);
#endif
	return static_cast<char_type*>(__builtin_memcpy(__s1, __s2, __n));
    1334:	f3 0f 6f 48 10       	movdqu xmm1,XMMWORD PTR [rax+0x10]
    1339:	0f 29 4c 24 50       	movaps XMMWORD PTR [rsp+0x50],xmm1
    133e:	e9 74 ff ff ff       	jmp    12b7 <main+0xf2>
    1343:	f3 0f 6f 40 10       	movdqu xmm0,XMMWORD PTR [rax+0x10]
    1348:	0f 29 44 24 50       	movaps XMMWORD PTR [rsp+0x50],xmm0
    134d:	eb a6                	jmp    12f5 <main+0x130>
    134f:	48 89 c7             	mov    rdi,rax
      return __ostream_insert(__os, __str.data(), __str.size());
    1352:	48 8b 54 24 48       	mov    rdx,QWORD PTR [rsp+0x48]
    1357:	48 8b 74 24 40       	mov    rsi,QWORD PTR [rsp+0x40]
    135c:	e8 0f fd ff ff       	call   1070 <std::basic_ostream<char, std::char_traits<char> >& std::__ostream_insert<char, std::char_traits<char> >(std::basic_ostream<char, std::char_traits<char> >&, char const*, long)@plt>
    1361:	48 89 c7             	mov    rdi,rax
    1364:	e8 d7 fc ff ff       	call   1040 <std::basic_ostream<char, std::char_traits<char> >& std::endl<char, std::char_traits<char> >(std::basic_ostream<char, std::char_traits<char> >&)@plt>
    1369:	eb 40                	jmp    13ab <main+0x1e6>
      { _M_dispose(); }
    136b:	48 89 c3             	mov    rbx,rax
    136e:	48 89 e7             	mov    rdi,rsp
    1371:	e8 1a fd ff ff       	call   1090 <std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_dispose()@plt>
    1376:	48 8d 7c 24 20       	lea    rdi,[rsp+0x20]
    137b:	e8 10 fd ff ff       	call   1090 <std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_dispose()@plt>
    1380:	48 8d 7c 24 60       	lea    rdi,[rsp+0x60]
    1385:	e8 06 fd ff ff       	call   1090 <std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_dispose()@plt>
      template<typename _Tp1>
	_GLIBCXX20_CONSTEXPR
	new_allocator(const new_allocator<_Tp1>&) _GLIBCXX_USE_NOEXCEPT { }

#if __cplusplus <= 201703L
      ~new_allocator() _GLIBCXX_USE_NOEXCEPT { }
    138a:	48 89 df             	mov    rdi,rbx
    138d:	e8 2e fd ff ff       	call   10c0 <_Unwind_Resume@plt>
    1392:	48 89 c3             	mov    rbx,rax
    1395:	eb df                	jmp    1376 <main+0x1b1>
    1397:	48 89 c3             	mov    rbx,rax
    139a:	48 8d 7c 24 40       	lea    rdi,[rsp+0x40]
    139f:	e8 ec fc ff ff       	call   1090 <std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_dispose()@plt>
    13a4:	eb da                	jmp    1380 <main+0x1bb>
    13a6:	48 89 c3             	mov    rbx,rax
    13a9:	eb d5                	jmp    1380 <main+0x1bb>
    13ab:	48 8d 7c 24 40       	lea    rdi,[rsp+0x40]
    13b0:	e8 db fc ff ff       	call   1090 <std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_dispose()@plt>
    13b5:	48 8d 7c 24 60       	lea    rdi,[rsp+0x60]
    13ba:	e8 d1 fc ff ff       	call   1090 <std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_dispose()@plt>
    return 0;
    13bf:	b8 00 00 00 00       	mov    eax,0x0
    13c4:	48 83 ec 80          	sub    rsp,0xffffffffffffff80
    13c8:	5b                   	pop    rbx
    13c9:	c3                   	ret    

Disassembly of section .fini:
