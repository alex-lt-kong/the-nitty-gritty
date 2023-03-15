
./string-demo.o:     file format elf64-x86-64


Disassembly of section .text:

0000000000000000 <__static_initialization_and_destruction_0(int, int)>:

void longStringNoSsoDemo(string prefix) {
    string longString = "Hello, world! This string is longer than 11/23 characters.";
    longString += prefix;
    cout << "longString: " << longString << endl;
   0:	cmp    edi,0x1
   3:	je     6 <__static_initialization_and_destruction_0(int, int)+0x6>
   5:	ret    
   6:	cmp    esi,0xffff
   c:	jne    5 <__static_initialization_and_destruction_0(int, int)+0x5>
   e:	sub    rsp,0x8
  extern wostream wclog;	/// Linked to standard error (buffered)
#endif
  //@}

  // For construction of filebuffers for cout, cin, cerr, clog et. al.
  static ios_base::Init __ioinit;
  12:	lea    rdi,[rip+0x0]        # 19 <__static_initialization_and_destruction_0(int, int)+0x19>
  19:	call   1e <__static_initialization_and_destruction_0(int, int)+0x1e>
  1e:	lea    rdx,[rip+0x0]        # 25 <__static_initialization_and_destruction_0(int, int)+0x25>
  25:	lea    rsi,[rip+0x0]        # 2c <__static_initialization_and_destruction_0(int, int)+0x2c>
  2c:	mov    rdi,QWORD PTR [rip+0x0]        # 33 <__static_initialization_and_destruction_0(int, int)+0x33>
  33:	call   38 <__static_initialization_and_destruction_0(int, int)+0x38>
  38:	add    rsp,0x8
  3c:	ret    

000000000000003d <longStringNoSsoDemo(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)>:
void longStringNoSsoDemo(string prefix) {
  3d:	push   rbp
  3e:	push   rbx
  3f:	sub    rsp,0x28
  43:	mov    rbx,rdi
#if __cplusplus < 201103L
	_Alloc_hider(pointer __dat, const _Alloc& __a = _Alloc())
	: allocator_type(__a), _M_p(__dat) { }
#else
	_Alloc_hider(pointer __dat, const _Alloc& __a)
	: allocator_type(__a), _M_p(__dat) { }
  46:	mov    rbp,rsp
  49:	lea    rax,[rsp+0x10]
  4e:	mov    QWORD PTR [rsp],rax
        void
        _M_construct_aux(_InIterator __beg, _InIterator __end,
			 std::__false_type)
	{
          typedef typename iterator_traits<_InIterator>::iterator_category _Tag;
          _M_construct(__beg, __end, _Tag());
  52:	lea    rdx,[rip+0x0]        # 59 <longStringNoSsoDemo(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)+0x1c>
  59:	lea    rsi,[rdx-0x3a]
  5d:	mov    rdi,rbp
  60:	call   65 <longStringNoSsoDemo(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)+0x28>
      // Capacity:
      ///  Returns the number of characters in the string, not including any
      ///  null-termination.
      size_type
      size() const _GLIBCXX_NOEXCEPT
      { return _M_string_length; }
  65:	mov    rdx,QWORD PTR [rbx+0x8]
      { return _M_dataplus._M_p; }
  69:	mov    rsi,QWORD PTR [rbx]
       *  @param __str  The string to append.
       *  @return  Reference to this string.
       */
      basic_string&
      append(const basic_string& __str)
      { return _M_append(__str._M_data(), __str.size()); }
  6c:	mov    rdi,rbp
  6f:	call   74 <longStringNoSsoDemo(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)+0x37>
    operator<<(basic_ostream<char, _Traits>& __out, const char* __s)
    {
      if (!__s)
	__out.setstate(ios_base::badbit);
      else
	__ostream_insert(__out, __s,
  74:	mov    edx,0xc
  79:	lea    rsi,[rip+0x0]        # 80 <longStringNoSsoDemo(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)+0x43>
  80:	mov    rdi,QWORD PTR [rip+0x0]        # 87 <longStringNoSsoDemo(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)+0x4a>
  87:	call   8c <longStringNoSsoDemo(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)+0x4f>
    operator<<(basic_ostream<_CharT, _Traits>& __os,
	       const basic_string<_CharT, _Traits, _Alloc>& __str)
    {
      // _GLIBCXX_RESOLVE_LIB_DEFECTS
      // 586. string inserter not a formatted function
      return __ostream_insert(__os, __str.data(), __str.size());
  8c:	mov    rdx,QWORD PTR [rsp+0x8]
  91:	mov    rsi,QWORD PTR [rsp]
  95:	mov    rdi,QWORD PTR [rip+0x0]        # 9c <longStringNoSsoDemo(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)+0x5f>
  9c:	call   a1 <longStringNoSsoDemo(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)+0x64>
  a1:	mov    rbx,rax
   *  for more on this subject.
  */
  template<typename _CharT, typename _Traits>
    inline basic_ostream<_CharT, _Traits>&
    endl(basic_ostream<_CharT, _Traits>& __os)
    { return flush(__os.put(__os.widen('\n'))); }
  a4:	mov    rax,QWORD PTR [rax]
  a7:	mov    rcx,rbx
  aa:	add    rcx,QWORD PTR [rax-0x18]
       *  Additional l10n notes are at
       *  http://gcc.gnu.org/onlinedocs/libstdc++/manual/localization.html
      */
      char_type
      widen(char __c) const
      { return __check_facet(_M_ctype).widen(__c); }
  ae:	mov    rbp,QWORD PTR [rcx+0xf0]
      if (!__f)
  b5:	test   rbp,rbp
  b8:	je     d2 <longStringNoSsoDemo(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)+0x95>
       *  @return  The converted character.
      */
      char_type
      widen(char __c) const
      {
	if (_M_widen_ok)
  ba:	cmp    BYTE PTR [rbp+0x38],0x0
  be:	je     ea <longStringNoSsoDemo(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)+0xad>
	  return _M_widen[static_cast<unsigned char>(__c)];
  c0:	movzx  esi,BYTE PTR [rbp+0x43]
  c4:	movsx  esi,sil
  c8:	mov    rdi,rbx
  cb:	call   d0 <longStringNoSsoDemo(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)+0x93>
  d0:	jmp    105 <longStringNoSsoDemo(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)+0xc8>
	__throw_bad_cast();
  d2:	call   d7 <longStringNoSsoDemo(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)+0x9a>
      { _M_dispose(); }
  d7:	mov    rbx,rax
  da:	mov    rdi,rsp
  dd:	call   e2 <longStringNoSsoDemo(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)+0xa5>
      template<typename _Tp1>
	_GLIBCXX20_CONSTEXPR
	new_allocator(const new_allocator<_Tp1>&) _GLIBCXX_USE_NOEXCEPT { }

#if __cplusplus <= 201703L
      ~new_allocator() _GLIBCXX_USE_NOEXCEPT { }
  e2:	mov    rdi,rbx
  e5:	call   ea <longStringNoSsoDemo(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)+0xad>
	this->_M_widen_init();
  ea:	mov    rdi,rbp
  ed:	call   f2 <longStringNoSsoDemo(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)+0xb5>
	return this->do_widen(__c);
  f2:	mov    rax,QWORD PTR [rbp+0x0]
  f6:	mov    esi,0xa
  fb:	mov    rdi,rbp
  fe:	call   QWORD PTR [rax+0x30]
 101:	mov    esi,eax
 103:	jmp    c4 <longStringNoSsoDemo(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)+0x87>
 105:	mov    rdi,rax
   *  This manipulator simply calls the stream's @c flush() member function.
  */
  template<typename _CharT, typename _Traits>
    inline basic_ostream<_CharT, _Traits>&
    flush(basic_ostream<_CharT, _Traits>& __os)
    { return __os.flush(); }
 108:	call   10d <longStringNoSsoDemo(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)+0xd0>
      { return _M_dataplus._M_p; }
 10d:	mov    rdi,QWORD PTR [rsp]
	if (!_M_is_local())
 111:	lea    rax,[rsp+0x10]
 116:	cmp    rdi,rax
 119:	je     129 <longStringNoSsoDemo(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)+0xec>
      { _Alloc_traits::deallocate(_M_get_allocator(), _M_data(), __size + 1); }
 11b:	mov    rax,QWORD PTR [rsp+0x10]
 120:	lea    rsi,[rax+0x1]
# endif
			      std::align_val_t(alignof(_Tp)));
	    return;
	  }
#endif
	::operator delete(__p
 124:	call   129 <longStringNoSsoDemo(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)+0xec>
 129:	add    rsp,0x28
 12d:	pop    rbx
 12e:	pop    rbp
 12f:	ret    

0000000000000130 <_GLOBAL__sub_I_string_demo.cpp>:
 130:	sub    rsp,0x8
 134:	mov    esi,0xffff
 139:	mov    edi,0x1
 13e:	call   0 <__static_initialization_and_destruction_0(int, int)>
 143:	add    rsp,0x8
 147:	ret    

Disassembly of section .text._ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPKcEEvT_S8_St20forward_iterator_tag:

0000000000000000 <void std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_construct<char const*>(char const*, char const*, std::forward_iterator_tag)>:
      }

  template<typename _CharT, typename _Traits, typename _Alloc>
    template<typename _InIterator>
      void
      basic_string<_CharT, _Traits, _Alloc>::
   0:	push   r12
   2:	push   rbp
   3:	push   rbx
   4:	sub    rsp,0x10
   8:	mov    rbp,rdi
   b:	mov    r12,rsi
   e:	mov    rbx,rdx
      _M_construct(_InIterator __beg, _InIterator __end,
		   std::forward_iterator_tag)
      {
	// NB: Not required, but considered best practice.
	if (__gnu_cxx::__is_null_pointer(__beg) && __beg != __end)
  11:	test   rsi,rsi
  14:	je     40 <void std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_construct<char const*>(char const*, char const*, std::forward_iterator_tag)+0x40>
               random_access_iterator_tag)
    {
      // concept requirements
      __glibcxx_function_requires(_RandomAccessIteratorConcept<
				  _RandomAccessIterator>)
      return __last - __first;
  16:	sub    rbx,r12
	  std::__throw_logic_error(__N("basic_string::"
				       "_M_construct null not valid"));

	size_type __dnew = static_cast<size_type>(std::distance(__beg, __end));
  19:	mov    QWORD PTR [rsp+0x8],rbx

	if (__dnew > size_type(_S_local_capacity))
  1e:	cmp    rbx,0xf
  22:	ja     51 <void std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_construct<char const*>(char const*, char const*, std::forward_iterator_tag)+0x51>
      { return _M_dataplus._M_p; }
  24:	mov    rdi,QWORD PTR [rbp+0x0]
	if (__n == 1)
  28:	cmp    rbx,0x1
  2c:	je     72 <void std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_construct<char const*>(char const*, char const*, std::forward_iterator_tag)+0x72>
      }

      static _GLIBCXX20_CONSTEXPR char_type*
      copy(char_type* __s1, const char_type* __s2, size_t __n)
      {
	if (__n == 0)
  2e:	test   rbx,rbx
  31:	je     79 <void std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_construct<char const*>(char const*, char const*, std::forward_iterator_tag)+0x79>
	  return __s1;
#ifdef __cpp_lib_is_constant_evaluated
	if (std::is_constant_evaluated())
	  return __gnu_cxx::char_traits<char_type>::copy(__s1, __s2, __n);
#endif
	return static_cast<char_type*>(__builtin_memcpy(__s1, __s2, __n));
  33:	mov    rdx,rbx
  36:	mov    rsi,r12
  39:	call   3e <void std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_construct<char const*>(char const*, char const*, std::forward_iterator_tag)+0x3e>
  3e:	jmp    79 <void std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_construct<char const*>(char const*, char const*, std::forward_iterator_tag)+0x79>
	if (__gnu_cxx::__is_null_pointer(__beg) && __beg != __end)
  40:	cmp    rsi,rdx
  43:	je     16 <void std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_construct<char const*>(char const*, char const*, std::forward_iterator_tag)+0x16>
	  std::__throw_logic_error(__N("basic_string::"
  45:	lea    rdi,[rip+0x0]        # 4c <void std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_construct<char const*>(char const*, char const*, std::forward_iterator_tag)+0x4c>
  4c:	call   51 <void std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_construct<char const*>(char const*, char const*, std::forward_iterator_tag)+0x51>
	  {
	    _M_data(_M_create(__dnew, size_type(0)));
  51:	lea    rsi,[rsp+0x8]
  56:	mov    edx,0x0
  5b:	mov    rdi,rbp
  5e:	call   63 <void std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_construct<char const*>(char const*, char const*, std::forward_iterator_tag)+0x63>
      { _M_dataplus._M_p = __p; }
  63:	mov    QWORD PTR [rbp+0x0],rax
	    _M_capacity(__dnew);
  67:	mov    rax,QWORD PTR [rsp+0x8]
      { _M_allocated_capacity = __capacity; }
  6c:	mov    QWORD PTR [rbp+0x10],rax
  70:	jmp    24 <void std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_M_construct<char const*>(char const*, char const*, std::forward_iterator_tag)+0x24>
      { __c1 = __c2; }
  72:	movzx  eax,BYTE PTR [r12]
  77:	mov    BYTE PTR [rdi],al
	  {
	    _M_dispose();
	    __throw_exception_again;
	  }

	_M_set_length(__dnew);
  79:	mov    rax,QWORD PTR [rsp+0x8]
      { _M_string_length = __length; }
  7e:	mov    QWORD PTR [rbp+0x8],rax
	traits_type::assign(_M_data()[__n], _CharT());
  82:	add    rax,QWORD PTR [rbp+0x0]
  86:	mov    BYTE PTR [rax],0x0
      }
  89:	add    rsp,0x10
  8d:	pop    rbx
  8e:	pop    rbp
  8f:	pop    r12
  91:	ret    
