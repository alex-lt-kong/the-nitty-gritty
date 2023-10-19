CC = gcc
CFLAGS = -O2 -Wall -Wc++-compat -pedantic -Wextra
LDFLAGS = -lrt -lpthread

RM = rm

# Directory to store the built targets
BUILD_DIR = ./build
TARGETS = shm-writer shm-reader
# The paths to the built targets
TARGET_PATHS = $(addprefix $(BUILD_DIR)/,$(TARGETS))

all: $(TARGET_PATHS)

$(BUILD_DIR)/%: %.c
	@mkdir -p $(BUILD_DIR)
# $@ is an automatic variable that represents target
# $< is an automatic variable that represents dependency
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<
	
.PHONY: clean
clean:
	$(RM) -r $(BUILD_DIR)
	