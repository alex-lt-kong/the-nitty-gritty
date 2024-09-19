CC = gcc
CXX = g++
LDFLAGS = -lfmt
CFLAGS = -O2 -Wall -pedantic -Wextra
INC = #-I/usr/include/glib-2.0 -I/usr/lib/x86_64-linux-gnu/glib-2.0/include/

TARGETS = 1_use 2_forward_iterator

# Directory to store the built targets
BUILD_DIR = build

# The paths to the built targets
TARGET_PATHS = $(addprefix $(BUILD_DIR)/,$(TARGETS))

all: prebuild $(TARGET_PATHS)

prebuild:
	@mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%: %.cpp
	$(CXX) $(INC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all prebuild clean