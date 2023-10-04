main:
	gcc -o shm_writer.exe shm_writer.c -O3 -Wall
	gcc -fPIC -shared -O3 -o shm_reader.so shm_reader.c

clean:
	rm *.out