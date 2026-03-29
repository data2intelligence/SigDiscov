# Intel MKL build (x86_64 only) -- multi-stage with selective lib copy
# NOTE: .so versions are tied to intel/oneapi-basekit:2024.0.1-devel-ubuntu22.04
FROM intel/oneapi-basekit:2024.0.1-devel-ubuntu22.04 AS builder

WORKDIR /build
COPY src/ src/
COPY Makefile .

RUN make BUILDDIR=. -j$(nproc)

# Discover exact shared lib names the binary needs (for maintainability)
RUN echo "=== Binary dependencies ===" && \
    ldd /build/morans_i_mkl | grep -E "mkl|iomp" && \
    echo "=== NEEDED entries ===" && \
    readelf -d /build/morans_i_mkl | grep NEEDED | grep -E "mkl|iomp"

# Collect only the .so files the binary actually links against
RUN mkdir -p /opt/intel/lib && \
    ldd /build/morans_i_mkl | grep -E "mkl|iomp" | awk '{print $3}' | \
    xargs -I{} cp -L {} /opt/intel/lib/

# Slim runtime stage
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/morans_i_mkl /usr/local/bin/morans_i_mkl
COPY --from=builder /opt/intel/lib/ /opt/intel/lib/

ENV LD_LIBRARY_PATH=/opt/intel/lib

# Verify binary runs
RUN morans_i_mkl --help > /dev/null 2>&1 || \
    (echo "ERROR: Binary fails to run -- missing libraries?" && ldd /usr/local/bin/morans_i_mkl && exit 1)

# Python tool
COPY tools/make_custom_w.py /usr/local/bin/make_custom_w.py
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

WORKDIR /data
ENTRYPOINT ["morans_i_mkl"]
CMD ["--help"]
