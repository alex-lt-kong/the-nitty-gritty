# Inter-Process Communication

## Approaches

 <table>
  <tr>
    <th>Method</th>
    <th>Short Description</th>
  </tr>
  <tr>
    <td>File</td>
    <td>A record stored on disk, or a record synthesized on demand by a file server, which can be accessed by multiple processes.</td>
  </tr>
  <tr>
    <td>Signal</td>
    <td>Signals are standardized messages sent to a running program to trigger specific behavior, such as quitting or error handling. They are a limited form of inter-process communication (IPC), typically used in Unix-like operating systems. </td>
  </tr>
  <tr>
    <td>Socket</td>
    <td>Data sent over a network interface. Implementations include TCP/UDP/etc</td>
  </tr>
  <tr>
    <td>Unix domain socket</td>
    <td>
    Similar to an internet socket, but all communication occurs within the kernel.
    Domain sockets use the file system as their address space.
    According to <a href="https://momjian.us/main/blogs/pgblog/2012.html#June_6_2012">this blog</a> from Postgres
    core developer Bruce Momjian, Unix socket can be 31% faster than TCP socket and 175% faster than TCP socket with SSL
    </td>
  </tr>
  <tr>
    <td>Message queue</td>
    <td>A data stream similar to a socket, but which usually preserves message boundaries. Examples include Apache kafka and RabbitMQ</td>
  </tr>
  <tr>
    <td>Anonymous pipe</td>
    <td>A unidirectional data channel using standard input and output. Data written to the write-end of the pipe is buffered by the operating system until it is read from the read-end of the pipe. Pipes can be created using the "|" character in many shells. </td>
  </tr>
  <tr>
    <td>Named pipe</td>
    <td>A pipe that is treated like a file. Instead of using standard input and output as with an anonymous pipe, processes write to and read from a named pipe, as if it were a regular file.</td>
  </tr>
  <tr>
    <td>Shared memory</td>
    <td>Multiple processes are given access to the same block of memory, which creates a shared buffer for the processes to communicate with each other.</td>
  </tr>
  <tr>
    <td>Message passing</td>
    <td>This concept includes a lot of different implementations, from gRPC to RESTful API</td>
  </tr>
  <tr>
    <td>Memory-mapped file</td>
    <td>A file mapped to RAM and can be modified by changing memory addresses directly instead of outputting to a stream. This shares the same benefits as a standard file.</td>
  </tr>
</table> 
https://en.wikipedia.org/wiki/Inter-process_communication

### Shared memory vs message-passing

https://www.geeksforgeeks.org/difference-between-shared-memory-model-and-message-passing-model-in-ipc/