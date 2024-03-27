CXX = g++
CXXFLAGS = -O2 -Wall -pedantic -Wextra -fsanitize=address

RM = rm

BUILD_DIR = ./build

all: unique_ptr.out value-ref-and-pointer-98.out value-ref-and-pointer-11.out

unique_ptr.out: unique_ptr.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) unique_ptr.cpp -o  $(BUILD_DIR)/unique_ptr $(CXXFLAGS)
value-ref-and-pointer-98.out: value-ref-and-pointer.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) value-ref-and-pointer.cpp -o  $(BUILD_DIR)/value-ref-and-pointer-98 -std=c++98 $(CXXFLAGS)
value-ref-and-pointer-11.out: value-ref-and-pointer.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) value-ref-and-pointer.cpp -o  $(BUILD_DIR)/value-ref-and-pointer-11 -std=c++11 $(CXXFLAGS)


.PHONY: clean
clean:
	$(RM) -r $(BUILD_DIR)
