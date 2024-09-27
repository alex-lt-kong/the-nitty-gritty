CC = gcc
CXX = g++
LDFLAGS = -lfmt
CFLAGS = -O2 -Wall -pedantic -Wextra
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
	$(CXX) $(INC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all prebuild clean