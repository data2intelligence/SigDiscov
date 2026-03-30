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
BINARY = morans_i_mkl
BUILDDIR ?= build
TARGET = $(BUILDDIR)/$(BINARY)

# Source layout
SRCDIR = src
VPATH  = $(SRCDIR)

# Source files (modular structure)
SOURCES = main.c \
          toy_example.c \
          morans_i_utils.c \
          morans_i_io_expression.c \
          morans_i_io_celltype.c \
          morans_i_io_weights.c \
          morans_i_io_results.c \
          morans_i_core.c \
          morans_i_spatial.c \
          morans_i_residual.c \
          morans_i_perm.c \
          morans_i_memory.c

HEADERS = $(SRCDIR)/morans_i_mkl.h $(SRCDIR)/morans_i_internal.h $(SRCDIR)/openblas_compat.h

OBJECTS = $(addprefix $(BUILDDIR)/,$(SOURCES:.c=.o))

# ============================================================
# Compiler Configuration
# ============================================================

# Detect build mode: GCC+OpenBLAS or Intel icx+MKL
ifdef USE_OPENBLAS
    # --- GCC + OpenBLAS Configuration ---
    CC ?= gcc
    CFLAGS = -O3 -fopenmp -Wall -Wextra -std=c99 -D_POSIX_C_SOURCE=200809L -D_GNU_SOURCE -g
    CFLAGS += -DUSE_OPENBLAS

    # OpenBLAS root: user override > pkg-config > module C_INCLUDE_PATH > /usr
    OPENBLAS_ROOT ?= $(shell pkg-config --variable=prefix openblas 2>/dev/null)
    ifeq ($(OPENBLAS_ROOT),)
        OPENBLAS_ROOT := $(shell echo "$$C_INCLUDE_PATH" | tr ':' '\n' | grep -i -m1 '[Oo]pen[Bb][Ll][Aa][Ss]' | sed 's|/include$$||')
    endif
    ifeq ($(OPENBLAS_ROOT),)
        OPENBLAS_ROOT := /usr
    endif
    INCLUDES = -I$(SRCDIR) -I$(OPENBLAS_ROOT)/include
    LDFLAGS = -fopenmp

    # OpenBLAS usually bundles LAPACKE; add -llapacke only if it doesn't
    HAS_LAPACKE_IN_OPENBLAS := $(shell nm -D $(OPENBLAS_ROOT)/lib/libopenblas.so 2>/dev/null | grep -q LAPACKE_dgetrf && echo yes)
    ifeq ($(HAS_LAPACKE_IN_OPENBLAS),yes)
        LIBS_LINK = -lopenblas -lpthread -lm -ldl
    else
        LIBS_LINK = -lopenblas -llapacke -lpthread -lm -ldl
    endif
    LIBS = $(LDFLAGS) -L$(OPENBLAS_ROOT)/lib $(LIBS_LINK)

    BUILD_INFO = GCC + OpenBLAS
else
    # --- Intel icx + MKL Configuration (default) ---
    CC = icx
    CFLAGS = -O3 -qopenmp -Wall -Wextra -std=c99 -D_POSIX_C_SOURCE=200809L -D_GNU_SOURCE -g

    # Find MKL (errors are deferred so 'make clean/help' always work)
    MKLROOT ?= $(shell dirname $$(dirname $$(which icx 2>/dev/null)) 2>/dev/null)/mkl/latest
    MKL_OK := yes
    ifeq ($(MKLROOT),)
        MKL_OK := no
    else ifeq ($(wildcard $(MKLROOT)/include/mkl.h),)
        MKL_OK := no
    endif

    INCLUDES = -I$(SRCDIR) -I$(MKLROOT)/include
    LDFLAGS = -qopenmp -L$(MKLROOT)/lib/intel64
    MKL_LIBS = -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl
    LIBS = $(LDFLAGS) $(MKL_LIBS)

    BUILD_INFO = Intel icx + MKL ($(MKLROOT))
endif

# ============================================================
# Build Rules
# ============================================================

# Defer MKL error to build targets only (so 'make clean/help' always work)
ifndef USE_OPENBLAS
ifneq ($(MKL_OK),yes)
NEED_MKL_ERROR := 1
endif
endif

# Default target
all: $(TARGET)

# Rule to build the target
$(TARGET): $(OBJECTS) | $(BUILDDIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LIBS)
	@echo "Build complete: $(TARGET) v$(VERSION) ($(BUILD_INFO))"

# Rule to compile object files (VPATH finds .c in src/)
$(BUILDDIR)/%.o: %.c $(HEADERS) Makefile | $(BUILDDIR)
ifdef NEED_MKL_ERROR
	$(error MKL not found (MKLROOT=$(MKLROOT)). Try: make CC=gcc USE_OPENBLAS=1)
endif
	$(CC) $(CFLAGS) $(VERSION_FLAG) $(INCLUDES) -c $< -o $@

# Create build directory
$(BUILDDIR):
	mkdir -p $(BUILDDIR)

# Rule to clean build artifacts
clean:
	rm -f $(TARGET) $(OBJECTS)
	@-rmdir $(BUILDDIR) 2>/dev/null || true
	@echo "Cleaned build artifacts."

# Install target
PREFIX ?= /usr/local
DESTDIR ?=
install: $(TARGET)
	mkdir -p $(DESTDIR)$(PREFIX)/bin
	cp $(TARGET) $(DESTDIR)$(PREFIX)/bin/$(BINARY)
	@echo "Installed $(BINARY) to $(DESTDIR)$(PREFIX)/bin/"

# Uninstall target
uninstall:
	rm -f $(DESTDIR)$(PREFIX)/bin/$(BINARY)
	@echo "Uninstalled $(BINARY) from $(DESTDIR)$(PREFIX)/bin/"

# Debug build with additional debug flags and no optimization
debug: CFLAGS_ORIG := $(CFLAGS)
debug: CFLAGS = $(filter-out -O3,$(CFLAGS_ORIG)) -O0 -DDEBUG_BUILD -g3
debug: clean all
	@echo "Debug build complete with CFLAGS: $(CFLAGS)"
debug: CFLAGS := $(CFLAGS_ORIG)

# Build a local OpenBLAS (no sudo required) then compile SigDiscov against it
LOCAL_OPENBLAS_DIR = $(CURDIR)/deps/openblas
deps-openblas:
	@echo "=== Building OpenBLAS locally (no sudo needed) ==="
	mkdir -p deps/_build
	cd deps/_build && \
	  if [ ! -d OpenBLAS ]; then \
	    git clone --depth 1 https://github.com/OpenMathLib/OpenBLAS.git; \
	  fi && \
	  cd OpenBLAS && \
	  make -j$$(nproc) NO_LAPACK=0 USE_OPENMP=1 PREFIX=$(LOCAL_OPENBLAS_DIR) && \
	  make PREFIX=$(LOCAL_OPENBLAS_DIR) install
	@echo "=== OpenBLAS installed to $(LOCAL_OPENBLAS_DIR) ==="
	@echo "Now run:  make CC=gcc USE_OPENBLAS=1 OPENBLAS_ROOT=$(LOCAL_OPENBLAS_DIR)"

# Phony targets
.PHONY: all clean install uninstall debug help test deps-openblas

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
	@echo "Build output goes to $(BUILDDIR)/ (override with BUILDDIR=<dir>)"
	@echo ""
	@echo "Makefile targets:"
	@echo "  all       - Build the program (default)"
	@echo "  clean     - Remove build artifacts"
	@echo "  install   - Install to $$(PREFIX)/bin (use 'make install PREFIX=/path')"
	@echo "  uninstall - Remove from $$(PREFIX)/bin"
	@echo "  debug     - Build with debug flags (-O0 -g3)"
	@echo "  test          - Submit regression tests to SLURM"
	@echo "  deps-openblas - Download and build OpenBLAS locally (no sudo)"
	@echo "  help          - Show this help message"
ifdef USE_OPENBLAS
	@echo ""
	@echo "Using GCC + OpenBLAS (OPENBLAS_ROOT=$(OPENBLAS_ROOT))"
else
	@echo ""
	@echo "Ensure Intel oneAPI environment (setvars.sh) is sourced before building."
	@echo "MKLROOT is currently set to: $(MKLROOT)"
endif
