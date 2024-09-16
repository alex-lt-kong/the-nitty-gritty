CC = gcc
LDFLAGS = -lpthread -lglib-2.0
CFLAGS = -O2 -Wall -pedantic -Wextra -Wc++-compat
INC = -I/usr/include/glib-2.0 -I/usr/lib/x86_64-linux-gnu/glib-2.0/include/

TARGETS = 1_hello-world 2_join-and-detach 3_with-parameters 4_multiple-threads \
          5_with-mutex 6_glibc-queue 7_condition-var
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