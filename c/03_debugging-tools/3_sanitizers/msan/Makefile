main:
	clang++ -fsanitize=memory -fsanitize-memory-track-origins -fPIE -pie -fno-omit-frame-pointer -g -O2 main.cpp -o main-msan.out
	clang++ -O2 main.cpp -o main.out

.PHONY: clean
clean:
	rm *.out