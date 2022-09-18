# List of undefined behaviors

 <table>
  <tr>
    <th>Description</th>
    <th>Example</th>
    <th>gcc's behavior</th>
  </tr>
  <tr>
    <td>Format specifier without argument</td>
    <td>`printf("%d");`/`snprintf(msg, MSG_BUF_SIZE, "[%s] [%d]")`</td>
    <td>random value from memory to stdout / null and 0 to stdout</td>
  </tr>
</table>
