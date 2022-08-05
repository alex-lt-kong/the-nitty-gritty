/ https://www.youtube.com/watch?v=4BZfwWbolPU
\p 9527

cub: {x*x*x}
cub2: {0N!x*x*x}
/ 0N! is basically used to print something to stdout.

worker: {[arg; my_callback_func] r:cub arg; (neg .z.w) (my_callback_func; r)}
/ .z.w is the handler server can use to make function calls to client