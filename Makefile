# Makefile for Moran's I calculation with Intel oneAPI and MKL
# Version: 1.2.1
# Compiler: Use icx (Intel LLVM-based compiler)
CC = icx
# Standard Flags: Optimization, OpenMP, Warnings, C standards, Debug symbols
# Added -D_GNU_SOURCE for strsep and other GNU extensions if used by headers like math.h for M_PI
# *POSIX*C_SOURCE=200809L for POSIX functions like strdup, getline
CFLAGS = -O3 -qopenmp -Wall -Wextra -std=c99 -D_POSIX_C_SOURCE=200809L -D_GNU_SOURCE -g
# Include MKL headers
MKLROOT ?= $(shell dirname $$(dirname $$(which icx)))/mkl/latest
# A more robust way to find MKLROOT if ONEAPI_ROOT is set:
# MKLROOT ?= $(ONEAPI_ROOT)/mkl/latest
ifeq ($(MKLROOT),)
    $(warning MKLROOT is not set or icx not found in PATH. Please source the Intel oneAPI setvars.sh script.)
    $(error MKLROOT not found. Halting.)
endif
ifeq ($(wildcard $(MKLROOT)/include/mkl.h),)
    $(warning MKLROOT ($(MKLROOT)) does not seem to contain MKL headers (mkl.h not found).)
    $(error MKL headers not found. Halting.)
endif
INCLUDES = -I$(MKLROOT)/include
# Linker Flags
LDFLAGS = -qopenmp -L$(MKLROOT)/lib/intel64
# For some systems, explicit path to libiomp5 might be needed if not in default search paths
# LDFLAGS += -L$(shell dirname $$(dirname $$(which icx)))/compiler/latest/linux/lib # Example path for libiomp5
# MKL Linking: Dynamic linking with Intel Threading Layer
MKL_LIBS = -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl
# Final linking flags
LIBS = $(LDFLAGS) $(MKL_LIBS)
# Version information - Updated to match header file
VERSION = 1.3.0
VERSION_FLAG = -DMORANS_I_MKL_VERSION=\"$(VERSION)\"
# Source files
SOURCES = main.c morans_i_mkl.c
HEADERS = morans_i_mkl.h
OBJECTS = $(SOURCES:.c=.o)
# Target executable name
TARGET = morans_i_mkl
# Default target
all: $(TARGET)
# Rule to build the target
$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) $^ -o $@ $(LIBS)
	@echo "Build complete: $(TARGET) v$(VERSION) (linked with MKL from $(MKLROOT))"
# Rule to compile object files
%.o: %.c $(HEADERS) Makefile
	$(CC) $(CFLAGS) $(VERSION_FLAG) $(INCLUDES) -c $< -o $@
# Dependencies (optional, compiler usually handles with -MMD -MP)
main.o: main.c morans_i_mkl.h
morans_i_mkl.o: morans_i_mkl.c morans_i_mkl.h
# Rule to clean build artifacts
clean:
	rm -f $(TARGET) $(OBJECTS) *.d # .d for dependency files if using -MMD
	@echo "Cleaned build artifacts."
# Install target (DESTDIR for staging, PREFIX for actual install root)
PREFIX ?= /usr/local
DESTDIR ?=
install: $(TARGET)
	mkdir -p $(DESTDIR)$(PREFIX)/bin
	cp $(TARGET) $(DESTDIR)$(PREFIX)/bin/
	@echo "Installed $(TARGET) to $(DESTDIR)$(PREFIX)/bin/"
# Uninstall target
uninstall:
	rm -f $(DESTDIR)$(PREFIX)/bin/$(TARGET)
	@echo "Uninstalled $(TARGET) from $(DESTDIR)$(PREFIX)/bin/"
# Debug build with additional debug flags and no optimization
debug: CFLAGS_ORIG := $(CFLAGS)
debug: CFLAGS = $(filter-out -O3,$(CFLAGS_ORIG)) -O0 -DDEBUG_BUILD -g3
debug: clean all
	@echo "Debug build complete with CFLAGS: $(CFLAGS)"
debug: CFLAGS := $(CFLAGS_ORIG) # Restore CFLAGS
# Phony targets
.PHONY: all clean install uninstall debug help
# Help target
help:
	@echo "Moran's I MKL - Version $(VERSION)"
	@echo "Makefile targets:"
	@echo "  all       - Build the program (default)"
	@echo "  clean     - Remove build artifacts"
	@echo "  install   - Install to $$(PREFIX)/bin (use 'make install PREFIX=/path/to/install')"
	@echo "  uninstall - Remove from $$(PREFIX)/bin"
	@echo "  debug     - Build with debug flags (-O0 -g3)"
	@echo "  help      - Show this help message"
	@echo ""
	@echo "Ensure Intel oneAPI environment (setvars.sh) is sourced before building."
	@echo "MKLROOT is currently set to: $(MKLROOT)"