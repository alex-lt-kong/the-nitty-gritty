CC = gcc
CFLAGS = -O2 -Wall -pedantic -Wextra -Wc++-compat

TARGETS = main
# Directory to store the built targets
BUILD_DIR = build
# The paths to the built targets
TARGET_PATHS = $(addprefix $(BUILD_DIR)/,$(TARGETS))

all: $(TARGET_PATHS)

$(BUILD_DIR)/%: %.c
	$(CC) $(INC) $(CFLAGS) -o $@ $<
 
.PHONY: clean
clean:
	rm  $(TARGET_PATHS)