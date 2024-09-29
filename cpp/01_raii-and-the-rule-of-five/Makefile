CXX = g++
CXXFLAGS = -Wall -pedantic -Wextra -fsanitize=address
INC = #-I/usr/include/glib-2.0 -I/usr/lib/x86_64-linux-gnu/glib-2.0/include/

RM = rm

SRC_FILES := $(wildcard *.cpp)
TARGETS := $(basename $(notdir $(SRC_FILES)))
BUILD_DIR = build
TARGET_PATHS = $(addprefix $(BUILD_DIR)/,$(TARGETS))

all: prebuild $(TARGET_PATHS)

prebuild:
	@mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%: %.cpp
	$(CXX) $(INC) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all prebuild clean