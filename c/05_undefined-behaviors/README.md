# Undefined behaviors

* While the general idea of undefined behavior is not difficult to understand, the exact wording may vary. In
[C99 standard](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1256.pdf), undefined behavior is defined as
"[u]ndefined behavior is otherwise indicated in this International Standard by the words "undefined behavior" or
by the omission of any explicit definition of behavior."
  * In the C community, undefined behavior may be humorously referred to as "nasal demons", after a comp.std.c post
  that explained undefined behavior as allowing the compiler to do anything it chooses, even "to make demons
  fly out of your nose".
  * It is common for programmers, even experienced ones, to rely on undefined behavior either by mistake, or
  simply because they are not well-versed in the rules of the language that can span hundreds of pages. This can result
  in bugs that are exposed when a different compiler or lead to security vulnerabilities in software.

* But if undefined behaviors are so bad, why don't we just define them?
  * Documenting an operation as undefined behavior allows compilers to assume that this operation will never happen
  in a conforming program. This gives the compiler more information about the code and this information can lead to
  more optimization opportunities. 
  * Simply put, the existence of undefined behaviors makes a language fast!

## List of undefined behaviors

 <table>
  <tr>
    <th>Description</th>
    <th>Example</th>
    <th>gcc's behavior</th>
  </tr>
  <tr>
    <td>Format specifier without argument</td>
    <td>
    <code>printf("%d");</code>
    <code>snprintf(msg, MSG_BUF_SIZE, "[%s] [%d]");</code>
    </td>
    <td>random value from memory to stdout / null and 0 to stdout</td>
  </tr>
  <tr>
    <td>Signed integer overflow</td>
    <td>
    <code>
      int a = 2147483600;
      for (int i=0; i<100; ++i) {<br>
        a++;<br>
        printf("%d\n", a);<br>
      }
    </code>
    </td>
    <td>
    -O1: 2147483647 + 1 becomes -2147483648<br />
    -O2/-O3: still observe the above, but the loop won't quit after
    <code>i</code> reaches 100. The program will be trapped in an infinite loop
    </td>
  </tr>
  <tr>
    <td>Initial value of <code>malloc()'ed</code> memory</td>
    <td>
    <code>
      arr = malloc(dim * 4);<br>
      for (int j = 0; j < dim; ++j) {<br>
        printf("%d, ", arr[j]);<br>
      }
      free(arr);
    </code>
    </td>
    <td>
      <code>malloc()</code> is documented in C standard and memory it returns are uninitialized.<br />
      According to
      <a href="https://stackoverflow.com/questions/8029584/why-does-malloc-initialize-the-values-to-0-in-gcc">this link</a>
      , behind <code>malloc()</code> there are two scenarios: it recycles <code>free()'ed</code> or requests more memory
      from the OS.<br />
      <li>if recycles memory from the already allocated one, memory blocks are returned by <code>malloc()</code> as
      they are, unchanged, i.e., programs can read the values that are already in those blocks assigned previously.</li>
      <li>if requests new memory from the OS, many OSes initialize all memory blocks to 0 as a
      security feature, which is not really related to C and <code>malloc()</code>.</li>
    </td>
  </tr>
  <tr>
    <td>Pass non-null-terminated C-string to <code>strlen()</code></td>
    <td>
    <code>
    char str[5] = "Hello";<br>
    strlen(str);
    </code>
    </td>
    <td>
    This case is a concrete example of array index-out-of-bound access, the result varies
    depending on where the pointer currently points to and whether a '\0' is near the bound of the c-string.
    </td>
  </tr>
</table>

## Seemingly undefined by actually well-defined behaviors

* `unsigned int` never overflow, for `unsigned int a = UINT_MAX;`, `a + 1` will be "wrapped around", i.e.,
`(a + 1) % UINT_MAX == 0`.
  * However, `unsigned int a = 2147483646 * 2147483646;` may lead to undefined behaviors since `2147483646 * 2147483646`
  could be considered as signed multiplication before the signed result is cast and assigned to `unsigned int a`.
  To make sure it is unsigned, rewrite it as `2147483646U * 2147483646U`.
