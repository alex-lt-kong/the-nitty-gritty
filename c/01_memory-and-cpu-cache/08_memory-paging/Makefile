CC = gcc
LDFLAGS = #-lpthread -lglib-2.0
CFLAGS = -O2 -Wall -pedantic -Wextra -Wc++-compat
INC = #-I/usr/include/glib-2.0 -I/usr/lib/x86_64-linux-gnu/glib-2.0/include/

TARGETS = main
BUILD_DIR = build
TARGET_PATHS = $(addprefix $(BUILD_DIR)/,$(TARGETS))

all: $(TARGET_PATHS)

$(BUILD_DIR)/%: %.c
	@mkdir -p $(BUILD_DIR)
	$(CC) $(INC) $(CFLAGS) -o $@ $< $(LDFLAGS)


.PHONY: clean
clean:
	rm  $(TARGET_PATHS)