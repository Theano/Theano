FROM nvdl.githost.io:4678/dgx/cuda:8.0-cudnn5.1-devel-ubuntu16.04--17.03

ENV THEANO_VERSION 0.9.0rc3
LABEL com.nvidia.theano.version="${THEANO_VERSION}"
ENV NVIDIA_THEANO_VERSION 17.03

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake \
        libopenblas-dev \
	cuda-cusolver-8-0 \
	cuda-cusolver-dev-8-0 \
        python-dev \
        apt-utils && \
    rm -rf /var/lib/apt/lists/*

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm -f get-pip.py

RUN pip install --upgrade --no-cache-dir pip setuptools wheel

ARG NVIDIA_BUILD_ID
ENV NVIDIA_BUILD_ID ${NVIDIA_BUILD_ID:-<unknown>}
LABEL com.nvidia.build.id="${NVIDIA_BUILD_ID}"
ARG NVIDIA_BUILD_REF
LABEL com.nvidia.build.ref="${NVIDIA_BUILD_REF}"

WORKDIR /opt/theano
COPY . .

RUN MAKEFLAGS="-j$(nproc)" \
    PREFIX=/usr/local \
    ./install.sh && \
    ldconfig

WORKDIR /workspace
COPY NVREADME.md README.md
COPY docker-examples docker-examples
COPY theanorc /workspace/.theanorc
ENV THEANORC /workspace/.theanorc
RUN chmod -R a+w /workspace

COPY nvidia_entrypoint.sh /usr/local/bin
ENTRYPOINT ["/usr/local/bin/nvidia_entrypoint.sh"]
