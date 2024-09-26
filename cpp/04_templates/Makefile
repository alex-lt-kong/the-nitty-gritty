CXX = clang++
CXXFLAGS = -O2 -Wall -pedantic -Wextra -g -Og -std=c++20

RM = rm

SRC_FILES := $(wildcard *.cpp)
TARGETS := $(basename $(notdir $(SRC_FILES)))
BUILD_DIR = build
TARGET_PATHS = $(addprefix $(BUILD_DIR)/,$(TARGETS))

all: prebuild $(TARGET_PATHS)

prebuild:
	@mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%: %.cpp
# $@ is an automatic variable that represents target
# $< is an automatic variable that represents dependency
	$(CXX) $(CXXFLAGS) -o $@ $<
	objdump --disassembler-options "intel" --disassemble=main --demangle --source $@ > $@.asm
	$(CXX) --version > $(BUILD_DIR)/cxx.version
	

clean:
	$(RM) -r $(BUILD_DIR)
		
.PHONY: all prebuild clean