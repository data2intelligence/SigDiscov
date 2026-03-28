# Intel MKL build (x86_64 only)
FROM intel/oneapi-basekit:2024.0.1-devel-ubuntu22.04

WORKDIR /build
COPY src/ src/
COPY Makefile .

RUN make -j$(nproc)

# Install Python for weight matrix generator
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*
COPY tools/make_custom_w.py /usr/local/bin/make_custom_w.py
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

RUN cp /build/morans_i_mkl /usr/local/bin/morans_i_mkl

WORKDIR /data
ENTRYPOINT ["morans_i_mkl"]
CMD ["--help"]
