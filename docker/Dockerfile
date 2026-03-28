# Intel MKL build (x86_64 only) -- multi-stage to reduce image size
FROM intel/oneapi-basekit:2024.0.1-devel-ubuntu22.04 AS builder

WORKDIR /build
COPY src/ src/
COPY Makefile .

RUN make -j$(nproc)

# Runtime stage -- only MKL runtime libs, not the full compiler toolkit
FROM ubuntu:22.04

# Install MKL runtime (shared libs only, no headers/compilers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    intel-oneapi-runtime-mkl-2024 \
    intel-oneapi-runtime-openmp-2024 \
    python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy only the binary
COPY --from=builder /build/morans_i_mkl /usr/local/bin/morans_i_mkl

# Python tool
COPY tools/make_custom_w.py /usr/local/bin/make_custom_w.py
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Set MKL library path
ENV LD_LIBRARY_PATH=/opt/intel/oneapi/lib:/opt/intel/oneapi/mkl/latest/lib/intel64:${LD_LIBRARY_PATH}

WORKDIR /data
ENTRYPOINT ["morans_i_mkl"]
CMD ["--help"]
