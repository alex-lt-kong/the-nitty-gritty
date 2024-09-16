CXX = clang++
CXXFLAGS = -O1 -g -Wall -pedantic -Wextra -fsanitize=address

RM = rm

# Directory to store the built targets
BUILD_DIR = ./build
TARGETS = base-constructor copy-constructor copy-elision lvalue-vs-rvalue move-constructor
# The paths to the built targets
TARGET_PATHS = $(addprefix $(BUILD_DIR)/,$(TARGETS))

all: $(TARGET_PATHS) move-constructor-noec

$(BUILD_DIR)/%: %.cpp
	@mkdir -p $(BUILD_DIR)
# $@ is an automatic variable that represents target
# $< is an automatic variable that represents dependency
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $<
move-constructor-noec: move-constructor.cpp
	$(CXX) $(CXXFLAGS) -fno-elide-constructors move-constructor.cpp -o $(BUILD_DIR)/move-constructor-noec

.PHONY: clean
clean:
	$(RM) -r $(BUILD_DIR)
