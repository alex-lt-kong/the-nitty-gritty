CXX = g++
LDFLAGS = -lopencv_cudacodec -lopencv_core -lopencv_videoio -lopencv_imgproc -lopencv_cudaimgproc -lopencv_cudaarithm  -lopencv_imgcodecs -lopencv_cudawarping
CXXFLAGS = -O2 -Wall -pedantic -Wextra
INC = -I/usr/local/include/opencv4/

TARGETS = cuda ffmpeg
# Directory to store the built targets
BUILD_DIR = build

# The paths to the built targets
TARGET_PATHS = $(addprefix $(BUILD_DIR)/,$(TARGETS))

all: $(TARGET_PATHS)

$(BUILD_DIR)/%: %.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(INC) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

.PHONY: clean
clean:
	rm  $(TARGET_PATHS)