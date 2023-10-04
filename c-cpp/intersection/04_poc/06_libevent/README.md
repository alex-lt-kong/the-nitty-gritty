# Libevent

* Event loop is a relatively popular design pattern, using by JavaScript etc.
Libevent is a common event loop library developed in C (of course).

* There is one concern though: JavaScript uses event loop design as well but
we can still corrupt data even if there is only one thread. The reason is that
JavaScript can, as I understand, "pause" the execution of one event and start
a second event, then resume the paused event. If the pause happens right
between lines that modify a shared object, it can corrupt my object.
    * According to [answers](https://stackoverflow.com/questions/76361184/does-libevent-process-two-events-concurrently-which-means-i-need-mutex?noredirect=1#comment134652836_76361184) from Stack Exchange, this doesn't appear to be
    the case of libevent.
    * This is demonstrated in [2_multiple-events.c](./2_multiple-events.c),
    a new task is never executed in the middle of another event.