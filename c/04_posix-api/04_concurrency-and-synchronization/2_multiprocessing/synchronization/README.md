# Mutex and Semaphore

## Mutex

## Semaphore

### Implementations

- There are a few implementations of semaphore, including:

    - more modern POSIX one (`sem_init()`/`sem_open()`, `sem_wait()`,
      `sem_post()`.
    - System V (SysV) one () (`semget()`, `semctl()`, `semop()`)

- According
  to [this link](https://stackoverflow.com/questions/368322/differences-between-system-v-and-posix-semaphores)
    - One marked difference between the System V and POSIX semaphore
      implementations is that in System V you can control how much the semaphore
      count can be increased or decreased; whereas in POSIX, the semaphore count
      is increased and decreased by 1.
    - POSIX semaphores do not allow manipulation of semaphore permissions,
      whereas System V semaphores allow you to change the permissions of
      semaphores to a subset of the original permission.
    - Initialization and creation of semaphores is atomic (from the user's
      perspective) in POSIX semaphores.
    - From a usage perspective, System V semaphores are clumsy, while POSIX
      semaphores are straight-forward
    - The scalability of POSIX semaphores (using unnamed semaphores) is much
      higher than System V semaphores. In a user/client
      scenario, where each user creates his own instances of a server, it would
      be better to use POSIX semaphores.
    - System V semaphores, when creating a semaphore object, creates an array of
      semaphores whereas POSIX semaphores create
      just one. Because of this feature, semaphore creation (memory
      footprint-wise) is costlier in System V semaphores when compared to POSIX
      semaphores.
    - It has been said that POSIX semaphore performance is better than System
      V-based semaphores.
    - POSIX semaphores provide a mechanism for process-wide semaphores rather
      than system-wide semaphores. So, if a developer forgets to close the
      semaphore, on process exit the semaphore is cleaned up. In simple terms,
      POSIX semaphores provide a mechanism for non-persistent semaphores.

### Performance

- According to
  this [link](http://ethan.tira-thompson.com/Semaphore_Lag_Time_Tests.html),
  time lag between setting a semaphore in one thread, and awaking each of `n`
  observing threads is on the order of 10 microseconds.  (13 - 50 us to be more
  accurate)

- On a GHz-class CPU, it means that it takes 13,000 - 50,000 CPU cycles to
  complete the semaphore signal

## Semaphore vs Mutex

- Mutex and Semaphore both provide synchronization services but they are not the
  same.

- Mutex is a locking mechanism whereas Semaphore is a signaling
  mechanism. <sup>[[Mutex vs.semaphore: What are the differences?](https://www.shiksha.com/online-courses/articles/mutex-vs-semaphore-what-are-the-differences/)]</sup>
  It means there is ownership associated with a mutex, and only the owner can
  release the lock (
  mutex). <sup>[[Mutex vs Semaphore](https://www.geeksforgeeks.org/mutex-vs-semaphore/)]</sup>

- A mutex object allows multiple process threads to access a single shared
  resource but only one at a time. On the other hand, semaphore allows multiple
  process threads to access the finite instance of the resource until
  available. <sup>[[Mutex vs.semaphore: What are the differences?](https://www.shiksha.com/online-courses/articles/mutex-vs-semaphore-what-are-the-differences/)]</sup>
-
- Semaphore is a signalling mechanism and a thread that is waiting on a
  semaphore can be signaled by another thread. This is different than a mutex as
  the mutex can be signaled only by the thread that called the wait
  function. <sup>[[Mutex and Semaphore](https://medium.com/@irfanhaydararman/mutex-and-semaphore-e223321ddd7c)]</sup>

- There are mainly two types of semaphores i.e. counting semaphores and binary
  semaphores. <sup>[[Difference Between Counting and Binary Semaphores](https://www.geeksforgeeks.org/difference-between-counting-and-binary-semaphores/)]</sup>

