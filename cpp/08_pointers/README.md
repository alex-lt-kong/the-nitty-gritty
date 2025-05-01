# Pointer Series

- Table of contents
    - [Ep. 1 - value, reference and pointer](1_value-ref-and-pointer.md)
    - [Ep. 2 - std::unique_ptr](2_unique-ptr.md)
    - [Ep. 3 - std::shared_ptr (and std::weak_ptr)](3_shared-ptr-and-weak-ptr.md)

- Cheatsheet
    - Valgrind:
        - Command:
          `valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose --log-file=valgrind-out.txt ./bin`
        - With valgrind Your program will run much slower (eg. 20 to 30 times)
          than normal, and use a lot more
          memory. [[1](https://valgrind.org/docs/manual/quick-start.html)].
        - Valgrind is in essence a virtual machine using just-in-time
          compilation techniques, including dynamic recompilation. Nothing from
          the original program ever gets run directly on the host processor.
          Instead, Valgrind first translates the program into a temporary,
          simpler form called intermediate representation (IR). After the
          conversion, Valgrind translates the IR back into machine code and lets
          the host processor run
          it. [[2](https://en.wikipedia.org/wiki/Valgrind)]