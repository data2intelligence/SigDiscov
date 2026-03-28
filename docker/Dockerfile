# Intel MKL build (x86_64 only)
# Uses the full oneAPI base image which includes MKL runtime libs.
# The image is large (~15GB) but avoids fragile runtime-only package setup.
FROM intel/oneapi-basekit:2024.0.1-devel-ubuntu22.04 AS builder

WORKDIR /build
COPY src/ src/
COPY Makefile .

RUN make -j$(nproc)

# Slim runtime stage -- copy binary + only the MKL/OpenMP shared libs it needs
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Copy binary
COPY --from=builder /build/morans_i_mkl /usr/local/bin/morans_i_mkl

# Copy only the MKL and OpenMP runtime libraries the binary links against
COPY --from=builder /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_intel_lp64.so.2 \
                    /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_intel_thread.so.2 \
                    /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_core.so.2 \
                    /opt/intel/lib/
COPY --from=builder /opt/intel/oneapi/compiler/latest/lib/libiomp5.so \
                    /opt/intel/lib/

ENV LD_LIBRARY_PATH=/opt/intel/lib

# Python tool
COPY tools/make_custom_w.py /usr/local/bin/make_custom_w.py
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

WORKDIR /data
ENTRYPOINT ["morans_i_mkl"]
CMD ["--help"]
