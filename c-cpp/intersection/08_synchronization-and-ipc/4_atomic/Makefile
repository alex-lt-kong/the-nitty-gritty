CC = gcc
LDFLAGS = -lpthread
CFLAGS = -O2 -Wall -pedantic -Wextra -Wc++-compat

RM = rm

TARGETS = atomic
# Directory to store the built targets
BUILD_DIR = build
# The paths to the built targets
TARGET_PATHS = $(addprefix $(BUILD_DIR)/,$(TARGETS))

all: $(TARGET_PATHS)

$(BUILD_DIR)/%: %.c
	@mkdir -p $(BUILD_DIR)
	$(CC) $(INC) $(CFLAGS) -o $@ $< $(LDFLAGS)

.PHONY: clean
clean:
	$(RM) -r $(BUILD_DIR)