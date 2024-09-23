CXX = g++
CXXFLAGS = -O2 -Wall -pedantic -Wextra -g -Og

RM = rm

# Directory to store the built targets
BUILD_DIR = ./build
TARGETS = 1_hello-world 2_dynamic-input 3_noinline 4_edge-cases 5_trait 6_parameter_pack
# The paths to the built targets
TARGET_PATHS = $(addprefix $(BUILD_DIR)/,$(TARGETS))

all: $(TARGET_PATHS)

$(BUILD_DIR)/%: %.cpp
	@mkdir -p $(BUILD_DIR)
# $@ is an automatic variable that represents target
# $< is an automatic variable that represents dependency
	$(CXX) $(CXXFLAGS) -o $@ $<
	objdump --disassembler-options "intel" --disassemble=main --demangle --source $@ > $@.asm
	$(CXX) --version > $(BUILD_DIR)/cxx.version
	
.PHONY: clean
clean:
	$(RM) -r $(BUILD_DIR)
	