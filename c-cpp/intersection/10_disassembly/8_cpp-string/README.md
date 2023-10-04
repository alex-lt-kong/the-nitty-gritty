# C++ String

* There is an old saing, `std::string` is `std::vector<char>` in disguise.
    * A `std::string` object usually occupies 3 words (i.e., 24 bytes on x64
    platforms) to store data pointer, size and capacity.

* One of the confusing part is due the fact that when writing C++
(instead of C) code, our code is unlikely do be directly compiled to machine
code. Instead, C++ compiler will "transform" our code, to the extent we no
longer recognize them, before doing the compilation.
    * Worse still, these transforms, mostly related to STL, are
    implementation-defined, there is no way we can be sure how exactly
    one implementation handles the gory details.
    * For example, on my computer, `std::string` is defined as
    `std::basic_string<char>` in `/usr/include/c++/10/string` at line 67.
    `std::basic_string<char>`, in turn, is defined in
    `/usr/include/C++/10/bits/basic_string.tcc`, with multiple overloaded
    constructors. (This is gcc-10's STL implementation).

* Initialization:
    ```asm
    11e2:	push   rbx
    11e3:	sub    rsp,0x50
    std::string myStr = "This is a comparatively long test string!";
    11e7:	lea    rdx,[rsp+0x4f]
    11ec:	lea    rdi,[rsp+0x20]
    11f1:	lea    rsi,[rip+0xe10]        # 2008 <_IO_stdin_used+0x8>
    11f8:	call   1080 <std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::basic_string(char const*, std::allocator<char> const&)@plt>>::basic_string(char const*, std::allocator<char> const&)@plt>
    ```
    * In this particular version, g++ chose `<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::basic_string(char const*, std::allocator<char> const&)@plt>` constructor, which is
    defined in `/usr/include/C++/10/bits/basic_string.tcc` at 677 - 681:
    ```C++
    template<typename _CharT, typename _Traits, typename _Alloc>
        basic_string<_CharT, _Traits, _Alloc>::
        basic_string(initializer_list<_CharT> __l, const _Alloc& __a)
        : _M_dataplus(_S_construct(__l.begin(), __l.end(), __a), __a)
        { }
    ```
    * The the rabbit hole goes deeper as this constructor does nothing more than
    calling `_M_dataplus(_S_construct(__l.begin(), __l.end(), __a), __a)`,
    which is defined in `/usr/include/C++/10/bits/basic_string.h` at 166:
    ```C++
        _Alloc_hider	_M_dataplus;
    ```
    in which `_Alloc_hider` is a struct:
    ```C++
        struct _Alloc_hider : allocator_type // TODO check __is_final
        {
        _Alloc_hider(pointer __dat, const _Alloc& __a)
        : allocator_type(__a), _M_p(__dat) { }
        _Alloc_hider(pointer __dat, _Alloc&& __a = _Alloc())
        : allocator_type(std::move(__a)), _M_p(__dat) { }

        pointer _M_p; // The actual data.
        };
    ```
    * Not being an STL implementor, perhaps we should stop digging deeper. The
    main take away: `[rsp+0x20]` is passed as the first parameter, this should be the
        pointer that stores data. 

* First `printf()`:
    ```asm
    120c:	mov    rcx,QWORD PTR [rsp+0x30]
    1211:	mov    rdx,QWORD PTR [rsp+0x28]
    1216:	lea    rdi,[rip+0xe15]        # 2032 <_IO_stdin_used+0x32>
    121d:	mov    eax,0x0
    1222:	call   1030 <printf@plt>
    1227:	jmp    1230 <main+0x4e>
    1229:	mov    ecx,0xf
    122e:	jmp    1211 <main+0x2f>
    ```
    * Judging from the calling convention of `printf()`:
        1. `lea    rdi,[rip+0xe15]`, load the string formatter
        ("%s, size: %lu, capacity: %lu\n") from `.rodata`
        1. `rsi` is always `[rip+0xe10]`, which stores the static string.
        1. `mov    rdx,QWORD PTR [rsp+0x28]`: this should store the size of the
        string.
        1. `mov    rcx,QWORD PTR [rsp+0x30]`: this should store the capacity of
        the string.


## References

* [Small String Optimizations][1]

[1]: https://cpp-optimizations.netlify.app/small_strings/ "Small String Optimizations"