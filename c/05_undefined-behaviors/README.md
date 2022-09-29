# List of undefined behaviors

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
      for (int i=0; i<100; ++i) {
        a++;
        printf("%d\n", a);
      }
    </code>
    </td>
    <td>
    -O1: 2147483647 + 1 becomes -2147483648<br />
    -O2/-O3: still observe the above, but the loop won't quit after i reaching 100, trapped in an infinite loop
    </td>
  </tr>
  <tr>
    <td>Initial value of malloc() memory</td>
    <td>
    <code>
      arr = malloc(dim * 4);
      for (int j = 0; j < dim; ++j) {
        printf("%d, ", arr[j]);
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
      <li>if recycles memory from already allocated one, memory blocks are returned by <code>malloc()</code> as they are, unchanged, i.e., program can read the values that are already in those blocks assigned previously.</li>
      <li>if requests new memory from the OS, many OSes initialize all memory blocks to 0 as a
      security feature, which is not really related to C and <code>malloc()</code>.</li>
    </td>
  </tr>
</table>

## Seemingly undefined by actually well-defined behaviors

* `unsigned int` never overflow, for `unsigned int a = UINT_MAX;`, `a + 1` will be "wrapped around", i.e.,
`(a + 1) % UINT_MAX == 0`.
  * However, `unsigned int a = 2147483646 * 2147483646;` may lead to undefined behaviors since `2147483646 * 2147483646`
  could be considered as signed multiplication before the signed result is cast and assigned to `unsigned int a`.
  To make sure it is unsigned, rewrite it as `2147483646U * 2147483646U`.