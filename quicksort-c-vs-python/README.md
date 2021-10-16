py ./quicksort.py 
>>> Pure python: 5.140 sec
>>> list.sort(): 0.537 sec
>>> Numpy.sort(): 0.091 sec

cgc quicksort.c -o quicksort -O3
./quicksort 
>>> 0.133 sec

