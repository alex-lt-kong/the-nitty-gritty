# Atomic

## C

- According to 8.1.1 of the
  [Intel® 64 and IA-32 Architectures Software Developer’s Manual](https://www.intel.com/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-software-developer-vol-3a-part-1-manual.pdf)
  and the 24.4.7.2 of the
  [The GNU C Library (glibc) manual](https://www.gnu.org/software/libc/manual/html_node/Atomic-Types.html)
  `int` without any qualifiers is mostly atomic already.

- `sig_atomic_t`, an integer type which can be accessed as an atomic entity
  even in the presence of asynchronous interrupts made by signals, is just
  an alias of `int` defined in `sig_atomic_t.h` and `types.h`:

  ```C
  typedef int __sig_atomic_t;
  typedef __sig_atomic_t sig_atomic_t;
  ```

- Unfortunately, glibc's document does not explicitly mention whether or not
  other integer types, such as `unsigned int`, `int32_t`, etc must also be atomic
  in the same manner.

  - But it is easy to envisage that on an x86-32 CPU, accessing `int64_t` might
    need more than one instruction and thus it is not atomic. Also noteworthy
    is that pointer types are guranteed to be as "atomic" as `int`, because
    pointers' size is always the size of `int`, regardless type of pointers (
    e.g., `char*`, `uint64_t*`, `some_big_struct*`)

  - Also, people from
    [this thread](https://stackoverflow.com/questions/77262636/atomic-operation-with-glibc-c-different-from-atomic?noredirect=1#comment136222958_77262636)
    believe that glibc's document is not that
    reliable at all, when refering to `int`, the document's editor may well
    mean to say `volatile int`

- One difficult question is, what does "atomic" is glibc's doc mean anyway?

  - It means something like this: for a variable `uint32_t a = 0`, I
    assign new value to it: `a = 0xFFFFFFFF;`. If I access `a`, it is
    impossible for me to read something like
    `0xFFFFFF00`/`0xFFFF0000`/`0x00FFFF00`/`0x0000FF00`/etc,
    the only possible results are either `0` or `0xFFFFFFFF`. There is never
    an intermediary state that can be observed by my program.

  - Well this definition is actually also debatable per
    [this post](https://stackoverflow.com/questions/77262636/atomic-operation-with-glibc-c-different-from-atomic?noredirect=1#comment136222958_77262636)
    . The argument of some users are even in the context of gcc+Linux+glibc,
    it is still hard to guarantee that "none of the built-in functions like
    memcpy (which the compiler may invent calls to) will try to copy smaller
    than int chunks on int-aligned arguments". So we would be better not
    relying on glibc's document just to be safe.

- Any other even more singificant confusion is that, even if glibc's doc is
  perfectly correct, it can still lead to incorrect use of `int`. Consider
  the following code snippet (that is originally from
  [here](https://lumian2015.github.io/lockFreeProgramming/c11-features-in-currency.html)):

  ```C
  _Atomic int acnt = 0;
  int cnt = 0;
  sig_atomic_t scnt = 0;
  void *adding(void *input) {
    for (int i = 0; i < 10000; i++) {
      ++acnt;
      ++cnt;
      ++scnt;
    }
    pthread_exit(NULL);
  }

  void test() {
    pthread_t tid[10];
    for (int i = 0; i < 10; i++)
      pthread_create(&tid[i], NULL, adding, NULL);
    for (int i = 0; i < 10; i++)
      pthread_join(tid[i], NULL);

    printf("the value of acnt is %d\n", acnt);
    printf("the value of cnt is %d\n", cnt);
    printf("the value of scnt is %d\n", scnt);
  }
  ```

  Only the output from `acnt` is correct:

  ```
  the value of acnt is 100000
  the value of cnt is 49723
  the value of scnt is 52380
  ```

- Why so? The devil is in the definition of "atomic", we does not only want
  the read and write to be "separately atomic", we need them to be atomic
  "as a whole". This is called "atomic read–modify–write" or "atomic RMW".
  While `++cnt` does look like one single instruction, it may actually a
  shorthand of `cnt = cnt + 1`, so we need to read `cnt`, modify it and write
  `cnt` back. These steps as a whole are not atomic, and thus the result
  isn't correct.
- But does it mean that if one C statement emits only one machine instruction,
  e.g., `add dword [num], 1`, then the operation must be atomic?

  - The answer is still negative... According to
    [this post](https://stackoverflow.com/questions/39393850/is-incrementing-an-int-effectively-atomic-in-specific-cases)

    > Memory-destination instructions (other than pure stores) are
    > read-modify-write operations that happen in multiple internal steps.
    > No architectural register is modified, but the CPU has to hold the data
    > internally while it sends it through its ALU. The actual register file
    > is only a small part of the data storage inside even the simplest CPU,
    > with latches holding outputs of one stage as inputs for another stage,
    > etc., etc.

    The very useful insight is that, even if there is one machine instruction,
    without explicitly asking the compiler/CPU to do so, the instruction may
    still well be non-atomic!

- So long story short, since C11, we should just use `_Atomic`, which
  handles all the dirty details for us once and for all
