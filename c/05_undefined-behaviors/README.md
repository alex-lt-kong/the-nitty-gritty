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
</table>

## Seemingly undefined by actually defined behaviors

* `unsigned int` never overflow, it is defined that ...