CC = gcc
LDFLAGS = -lpthread
CFLAGS = -O2 -Wall -pedantic -Wextra -Wc++-compat

TARGETS = atomic-c
# Directory to store the built targets
BUILD_DIR = build
# The paths to the built targets
TARGET_PATHS = $(addprefix $(BUILD_DIR)/,$(TARGETS))

all: $(TARGET_PATHS)

$(BUILD_DIR)/%: %.c
	$(CC) $(INC) $(CFLAGS) -o $@ $< $(LDFLAGS)

.PHONY: clean
clean:
	rm  $(TARGET_PATHS)