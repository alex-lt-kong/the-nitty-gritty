
./main.out:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .text:

00000000000011e2 <main>:
#include <string>
#include <iostream>

int main(void) {
    11e2:	push   rbx
    11e3:	sub    rsp,0x50
    std::string myStr = "This is a comparatively long test string!";
    11e7:	lea    rdx,[rsp+0x4f]
    11ec:	lea    rdi,[rsp+0x20]
    11f1:	lea    rsi,[rip+0xe10]        # 2008 <_IO_stdin_used+0x8>
    11f8:	call   1080 <std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::basic_string(char const*, std::allocator<char> const&)@plt>
      _M_length(size_type __length)
      { _M_string_length = __length; }

      pointer
      _M_data() const
      { return _M_dataplus._M_p; }
    11fd:	mov    rsi,QWORD PTR [rsp+0x20]
       *  before needing to allocate more memory.
       */
      size_type
      capacity() const _GLIBCXX_NOEXCEPT
      {
	return _M_is_local() ? size_type(_S_local_capacity)
    1202:	lea    rax,[rsp+0x30]
    1207:	cmp    rsi,rax
    120a:	je     1229 <main+0x47>
    120c:	mov    rcx,QWORD PTR [rsp+0x30]
    printf("%s, size: %lu, capacity: %lu\n",
    1211:	mov    rdx,QWORD PTR [rsp+0x28]
    1216:	lea    rdi,[rip+0xe15]        # 2032 <_IO_stdin_used+0x32>
    121d:	mov    eax,0x0
    1222:	call   1030 <printf@plt>
    1227:	jmp    1230 <main+0x4e>
    1229:	mov    ecx,0xf
    122e:	jmp    1211 <main+0x2f>
	: allocator_type(std::move(__a)), _M_p(__dat) { }
    1230:	mov    rsi,rsp
    1233:	lea    rax,[rsp+0x10]
    1238:	mov    QWORD PTR [rsp],rax
      { _M_string_length = __length; }
    123c:	mov    QWORD PTR [rsp+0x8],0x0
      using comparison_category = strong_ordering;
#endif

      static _GLIBCXX17_CONSTEXPR void
      assign(char_type& __c1, const char_type& __c2) _GLIBCXX_NOEXCEPT
      { __c1 = __c2; }
    1245:	mov    BYTE PTR [rsp+0x10],0x0
        myStr.c_str(), myStr.size(), myStr.capacity());
    std::string line;
    std::getline(std::cin, line);
    124a:	lea    rdi,[rip+0x2e2f]        # 4080 <std::cin@@GLIBCXX_3.4>
    1251:	call   1040 <std::basic_istream<char, std::char_traits<char> >& std::getline<char, std::char_traits<char>, std::allocator<char> >(std::basic_istream<char, std::char_traits<char> >&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >&)@plt>
       *  @param __str  The string to append.
       *  @return  Reference to this string.
       */
      basic_string&
      append(const basic_string& __str)
      { return _M_append(__str._M_data(), __str.size()); }
    1256:	lea    rdi,[rsp+0x20]
    125b:	mov    rdx,QWORD PTR [rsp+0x8]
    1260:	mov    rsi,QWORD PTR [rsp]
    1264:	call   1050 <std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_append(char const*, unsigned long)@plt>
      { return _M_dataplus._M_p; }
    1269:	mov    rsi,QWORD PTR [rsp+0x20]
	return _M_is_local() ? size_type(_S_local_capacity)
    126e:	lea    rax,[rsp+0x30]
    1273:	cmp    rsi,rax
    1276:	je     1295 <main+0xb3>
    1278:	mov    rcx,QWORD PTR [rsp+0x30]
    myStr += line;
    printf("%s, size: %lu, capacity: %lu\n",
    127d:	mov    rdx,QWORD PTR [rsp+0x28]
    1282:	lea    rdi,[rip+0xda9]        # 2032 <_IO_stdin_used+0x32>
    1289:	mov    eax,0x0
    128e:	call   1030 <printf@plt>
    1293:	jmp    12be <main+0xdc>
    1295:	mov    ecx,0xf
    129a:	jmp    127d <main+0x9b>
      { _M_dispose(); }
    129c:	mov    rbx,rax
    129f:	mov    rdi,rsp
    12a2:	call   1070 <std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_dispose()@plt>
    12a7:	lea    rdi,[rsp+0x20]
    12ac:	call   1070 <std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_dispose()@plt>
      template<typename _Tp1>
	_GLIBCXX20_CONSTEXPR
	new_allocator(const new_allocator<_Tp1>&) _GLIBCXX_USE_NOEXCEPT { }

#if __cplusplus <= 201703L
      ~new_allocator() _GLIBCXX_USE_NOEXCEPT { }
    12b1:	mov    rdi,rbx
    12b4:	call   10a0 <_Unwind_Resume@plt>
    12b9:	mov    rbx,rax
    12bc:	jmp    12a7 <main+0xc5>
    12be:	mov    rdi,rsp
    12c1:	call   1070 <std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_dispose()@plt>
    12c6:	lea    rdi,[rsp+0x20]
    12cb:	call   1070 <std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_dispose()@plt>
        myStr.c_str(), myStr.size(), myStr.capacity());
    return 0;
    12d0:	mov    eax,0x0
    12d5:	add    rsp,0x50
    12d9:	pop    rbx
    12da:	ret    

Disassembly of section .fini:
