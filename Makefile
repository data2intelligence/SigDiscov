# Makefile for Moran's I calculation
# Version: 1.3.0
#
# Supports two build configurations:
#   1. Intel icx + MKL (default, recommended for HPC)
#   2. GCC + OpenBLAS (portability fallback)
#
# Usage:
#   make                      # Build with Intel icx + MKL (default)
#   make CC=gcc USE_OPENBLAS=1 # Build with GCC + OpenBLAS
#   make debug                # Debug build
#   make clean                # Remove build artifacts
#   make help                 # Show all targets

# Version information
VERSION = 1.3.0
VERSION_FLAG = -DMORANS_I_MKL_VERSION=\"$(VERSION)\"

# Target executable name
TARGET = morans_i_mkl

# Source files (modular structure)
SOURCES = main.c \
          morans_i_utils.c \
          morans_i_io.c \
          morans_i_core.c \
          morans_i_spatial.c \
          morans_i_residual.c \
          morans_i_perm.c \
          morans_i_memory.c

HEADERS = morans_i_mkl.h morans_i_internal.h openblas_compat.h

OBJECTS = $(SOURCES:.c=.o)

# ============================================================
# Compiler Configuration
# ============================================================

# Detect build mode: GCC+OpenBLAS or Intel icx+MKL
ifdef USE_OPENBLAS
    # --- GCC + OpenBLAS Configuration ---
    CC ?= gcc
    CFLAGS = -O3 -fopenmp -Wall -Wextra -std=c99 -D_POSIX_C_SOURCE=200809L -D_GNU_SOURCE -g
    CFLAGS += -DUSE_OPENBLAS

    # OpenBLAS includes and libs
    OPENBLAS_ROOT ?= /usr
    INCLUDES = -I$(OPENBLAS_ROOT)/include
    LDFLAGS = -fopenmp
    LIBS_LINK = -lopenblas -lpthread -lm -ldl
    LIBS = $(LDFLAGS) -L$(OPENBLAS_ROOT)/lib $(LIBS_LINK)

    BUILD_INFO = GCC + OpenBLAS
else
    # --- Intel icx + MKL Configuration (default) ---
    CC = icx
    CFLAGS = -O3 -qopenmp -Wall -Wextra -std=c99 -D_POSIX_C_SOURCE=200809L -D_GNU_SOURCE -g

    # Find MKL
    MKLROOT ?= $(shell dirname $$(dirname $$(which icx 2>/dev/null)) 2>/dev/null)/mkl/latest
    ifeq ($(MKLROOT),)
        $(warning MKLROOT is not set or icx not found in PATH. Please source the Intel oneAPI setvars.sh script.)
        $(error MKLROOT not found. Halting. Try: make CC=gcc USE_OPENBLAS=1)
    endif
    ifeq ($(wildcard $(MKLROOT)/include/mkl.h),)
        $(warning MKLROOT ($(MKLROOT)) does not seem to contain MKL headers (mkl.h not found).)
        $(error MKL headers not found. Halting. Try: make CC=gcc USE_OPENBLAS=1)
    endif

    INCLUDES = -I$(MKLROOT)/include
    LDFLAGS = -qopenmp -L$(MKLROOT)/lib/intel64
    MKL_LIBS = -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl
    LIBS = $(LDFLAGS) $(MKL_LIBS)

    BUILD_INFO = Intel icx + MKL ($(MKLROOT))
endif

# ============================================================
# Build Rules
# ============================================================

# Default target
all: $(TARGET)

# Rule to build the target
$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) $^ -o $@ $(LIBS)
	@echo "Build complete: $(TARGET) v$(VERSION) ($(BUILD_INFO))"

# Rule to compile object files
%.o: %.c $(HEADERS) Makefile
	$(CC) $(CFLAGS) $(VERSION_FLAG) $(INCLUDES) -c $< -o $@

# Rule to clean build artifacts
clean:
	rm -f $(TARGET) $(OBJECTS) *.d
	@echo "Cleaned build artifacts."

# Install target
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
debug: CFLAGS := $(CFLAGS_ORIG)

# Phony targets
.PHONY: all clean install uninstall debug help test

# Test target (submits via SLURM)
test:
	@echo "Submitting test job to SLURM..."
	sbatch tests/run_tests_slurm.sh

# Help target
help:
	@echo "Moran's I MKL - Version $(VERSION)"
	@echo ""
	@echo "Build configurations:"
	@echo "  make                        - Build with Intel icx + MKL (default)"
	@echo "  make CC=gcc USE_OPENBLAS=1  - Build with GCC + OpenBLAS"
	@echo ""
	@echo "Makefile targets:"
	@echo "  all       - Build the program (default)"
	@echo "  clean     - Remove build artifacts"
	@echo "  install   - Install to $$(PREFIX)/bin (use 'make install PREFIX=/path')"
	@echo "  uninstall - Remove from $$(PREFIX)/bin"
	@echo "  debug     - Build with debug flags (-O0 -g3)"
	@echo "  test      - Submit regression tests to SLURM"
	@echo "  help      - Show this help message"
ifdef USE_OPENBLAS
	@echo ""
	@echo "Using GCC + OpenBLAS (OPENBLAS_ROOT=$(OPENBLAS_ROOT))"
else
	@echo ""
	@echo "Ensure Intel oneAPI environment (setvars.sh) is sourced before building."
	@echo "MKLROOT is currently set to: $(MKLROOT)"
endif
