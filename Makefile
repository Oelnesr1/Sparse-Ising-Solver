# Define variables for compiler, flags, and directories
NVCC = nvcc
NVCC_FLAGS = -O2 -g
G++ = g++
G++_FLAGS = -std=c++11 -O2 -g
SRC_DIR = ./src
BUILD_DIR = ./build

# Collect source files
CU_SOURCES = $(wildcard $(SRC_DIR)/*.cu)
CPP_SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS = $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(CU_SOURCES)) \
          $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(CPP_SOURCES))

# Main target
program: $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) -o $(BUILD_DIR)/$@ $^

# Build CUDA object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

# Build C++ object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(G++) $(G++_FLAGS) -c -o $@ $<

# Clean target
.PHONY: clean
clean:
	rm -f $(BUILD_DIR)/*.o $(BUILD_DIR)/program
