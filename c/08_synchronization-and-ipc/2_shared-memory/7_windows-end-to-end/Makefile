COMMON_OPT = -Wall -O3

main: shm_writer.exe shm_reader.so

shm_writer.exe: shm_writer.c col_parser.c common.o
	gcc -o shm_writer.exe shm_writer.c common.o $(COMMON_OPT)
shm_reader.so: shm_reader.c common.o
	gcc -fPIC -shared -o shm_reader.so shm_reader.c common.o $(COMMON_OPT)
common.o: common.c common.h
	gcc -c common.c $(COMMON_OPT)

clean:
	rm *.exe *.so *.out