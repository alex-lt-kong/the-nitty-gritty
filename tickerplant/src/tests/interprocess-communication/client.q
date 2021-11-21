
my_h:hopen `:localhost:9527
/ if hopen connects to the remote server successfully, it returns a function
/ and we store the function to an arbitrary variable h

my_h (`cub, 3)
my_h (`cub2, 4)
show_result: {-1"Result from server:\n";show x}
(neg my_h) (`worker, 4, `show_result)
/ switch the sign of a handler makes it a asynchronous call
