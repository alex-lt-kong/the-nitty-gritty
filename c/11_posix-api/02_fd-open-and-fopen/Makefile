#main: 1_open.c 2_stdio.c
#	gcc 1_open.c -o 1_open.out
#	gcc 2_stdio.c -o 2_stdio.out
#	gcc 3_write-to-stdin.c -o 3_write-to-stdin.out
#	gcc 4_pseudo-device-io.c -o 4_pseudo-device-io.out

CC = gcc
CFLAGS = -O2 -Wall -pedantic -Wextra

RM = rm

SRC_FILES := $(wildcard *.c)
TARGETS := $(basename $(notdir $(SRC_FILES)))
BUILD_DIR = build
TARGET_PATHS = $(addprefix $(BUILD_DIR)/,$(TARGETS))

all: prebuild $(TARGET_PATHS)

prebuild:
	@mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%: %.c
# $@ is an automatic variable that represents target
# $< is an automatic variable that represents dependency
	$(CXX) $(CXXFLAGS) -o $@ $<
	

clean:
	$(RM) -r $(BUILD_DIR)
		
.PHONY: all prebuild clean