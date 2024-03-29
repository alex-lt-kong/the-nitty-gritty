# Multiple Semaphores

* If there is only one memory writer but multiple memory readers, it seems that
we have to rely on multiple semaphores to properly synchronize data.

* But if there are quite a lot of semaphores (~100 let's say), will it penalize
the performance greatly?

* Test results appear to show that this isn't the case--the performace remains
more or less the same with 128 semaphores initialized:

  * sample shm_write:
    ```
    $ ./shm_writer.out
    shared mem address: 0x7f9959f41000 [0..4095]
    sem_wait()'ed, press any key to memset() then sem_post()
    ```
  * Using `semaphores[127]`:
    ```
    $ ./shm_reader.out 127
    sem_wait()'ing
    sem_wait()'ed at 1665763779.760564245

    ========== Shared memory buffer BEGIN at 1665763779.760494245 ==========
    YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY
    ......3584 bytes in shared memory truncated......
    YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY
    ========== Shared memory buffer END at 1665763779.760514545 ==========
    ```
  * Using `semaphores[92]`:
    ```
    ./shm_reader.out 92
    sem_wait()'ing
    sem_wait()'ed at 1665763933.210126364

    ========== Shared memory buffer BEGIN at 1665763933.210038363 ==========
    YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY
    ......3584 bytes in shared memory truncated......
    YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY
    ========== Shared memory buffer END at 1665763933.210065764 ==========
    ```
  * Using `semaphores[23]`:
    ```
    $ ./shm_reader.out 23
    sem_wait()'ing
    sem_wait()'ed at 1665763968.643438516

    ========== Shared memory buffer BEGIN at 1665763968.643367515 ==========
    YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY
    ......3584 bytes in shared memory truncated......
    YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY
    ========== Shared memory buffer END at 1665763968.643385616 ==========
    ```
  * Using `semaphores[0]`:
    ```
    ./shm_reader.out 0
    sem_wait()'ing
    sem_wait()'ed at 1665764002.627272745

    ========== Shared memory buffer BEGIN at 1665764002.627202445 ==========
    YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY
    ......3584 bytes in shared memory truncated......
    YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY
    ========== Shared memory buffer END at 1665764002.627220045 ==========
    ```

* It still takes ~50 us for `shm_reader.out` to be signaled.