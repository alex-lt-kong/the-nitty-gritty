# Semaphore

* Mutex and Semaphore both provide synchronization services but they are not the same.

* Mutex is a locking mechanism whereas Semaphore is a signaling mechanism

* A mutex object allows multiple process threads to access a single shared resource
but only one at a time. On the other hand, semaphore allows multiple process threads 
to access the finite instance of the resource until available.

* Mutex is a mutual exclusion object that synchronizes access to a resource. It is created
with a unique name at the start of a program. The Mutex is a locking mechanism that makes
sure only one thread can acquire the Mutex at a time and enter the critical section.

* Semaphore is a signalling mechanism and a thread that is waiting on a semaphore
can be signaled by another thread. This is different than a mutex as the mutex can be
signaled only by the thread that called the wait function.

* There are mainly two types of semaphores i.e. counting semaphores and binary semaphores.

* Counting Semaphores are integer value semaphores and have an unrestricted value domain. These semaphores are used to coordinate the resource access, where the semaphore count is the number of available resources.