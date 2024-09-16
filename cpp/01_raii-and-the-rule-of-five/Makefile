CXX = g++
CXXFLAGS = -Wall -pedantic -Wextra -fsanitize=address

RM = rm

# Directory to store the built targets
BUILD_DIR = ./build
TARGETS = 01_naive-example 02_raw-pointers-with-raii 03_unique-ptr-with-raii
# The paths to the built targets
TARGET_PATHS = $(addprefix $(BUILD_DIR)/,$(TARGETS))

all: $(TARGET_PATHS)

$(BUILD_DIR)/%: %.cpp
	@mkdir -p $(BUILD_DIR)
# $@ is an automatic variable that represents target
# $< is an automatic variable that represents dependency
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $<
	
.PHONY: clean
clean:
	$(RM) -r $(BUILD_DIR)
