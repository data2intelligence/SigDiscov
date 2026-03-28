# Intel MKL build (x86_64 only)
# Multi-stage: build with full oneAPI toolkit, run with minimal runtime
FROM intel/oneapi-basekit:2024.0.1-devel-ubuntu22.04 AS builder

WORKDIR /build
COPY src/ src/
COPY Makefile .

RUN make -j$(nproc)

# Runtime stage -- slim image with only MKL runtime libs
FROM intel/oneapi-runtime:2024.0.1-devel-ubuntu22.04

COPY --from=builder /build/morans_i_mkl /usr/local/bin/morans_i_mkl

# Copy Python tool and deps
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*
COPY tools/make_custom_w.py /usr/local/bin/make_custom_w.py
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

WORKDIR /data
ENTRYPOINT ["morans_i_mkl"]
CMD ["--help"]
