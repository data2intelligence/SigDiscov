# Makefile for Moran's I calculation with Intel oneAPI and MKL
# Compiler: Use icx (Intel LLVM-based compiler)
CC = icx
# Standard Flags: Optimization, OpenMP, Warnings
CFLAGS = -O3 -qopenmp -Wall -Wextra -g
# Include MKL headers
MKLROOT ?= $(MKL_ROOT)
ifeq ($(MKLROOT),)
    $(error MKLROOT is not set. Please source the Intel oneAPI setvars.sh script first)
endif
INCLUDES = -I$(MKLROOT)/include
# Linker Flags
LDFLAGS = -qopenmp -L$(MKLROOT)/lib/intel64
# MKL Linking: Dynamic linking
MKL_LIBS = -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl
# Final linking flags
LIBS = $(LDFLAGS) $(MKL_LIBS)
# Source files
SOURCES = main.c morans_i_mkl.c
OBJECTS = $(SOURCES:.c=.o)
# Target executable name
TARGET = morans_i_mkl

# Default target
all: $(TARGET)

# Rule to build the target
$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)
	@echo "Build complete: $(TARGET)"

# Rule to compile object files
%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Dependencies
main.o: main.c morans_i_mkl.h
morans_i_mkl.o: morans_i_mkl.c morans_i_mkl.h

# Rule to clean build artifacts
clean:
	rm -f $(TARGET) $(OBJECTS)
	@echo "Cleaned build artifacts."

# Install target (optional)
install: $(TARGET)
	mkdir -p $(DESTDIR)/usr/local/bin
	cp $(TARGET) $(DESTDIR)/usr/local/bin/

# Uninstall target (optional)
uninstall:
	rm -f $(DESTDIR)/usr/local/bin/$(TARGET)

# Phony targets
.PHONY: all clean install uninstall