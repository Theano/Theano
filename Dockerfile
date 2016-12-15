FROM nvdl.githost.io:4678/dgx/cuda:8.0-cudnn5-devel-ubuntu14.04
MAINTAINER NVIDIA CORPORATION <cudatools@nvidia.com>

ENV THEANO_VERSION 0.8.X
LABEL com.nvidia.theano.version="0.8.X"
ENV NVIDIA_THEANO_VERSION 16.12

ARG NVIDIA_BUILD_ID
ENV NVIDIA_BUILD_ID ${NVIDIA_BUILD_ID:-<unknown>}
LABEL com.nvidia.build.id="${NVIDIA_BUILD_ID}"
ARG NVIDIA_BUILD_REF
LABEL com.nvidia.build.ref="${NVIDIA_BUILD_REF}"

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
		ca-certificates \
		curl \
        libopenblas-dev \
        python-dev && \
    rm -rf /var/lib/apt/lists/*

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

WORKDIR /opt/theano
COPY . .

RUN pip install --upgrade --no-cache-dir pip setuptools wheel && \
    cat requirement-rtd.txt | xargs -n1 pip install --no-cache-dir

RUN pip install -e .

WORKDIR /workspace
COPY NVREADME.md README.md
COPY benchmark benchmark
COPY theanorc /workspace/.theanorc
ENV THEANORC /workspace/.theanorc
RUN chmod -R a+w /workspace

COPY nvidia_entrypoint.sh /usr/local/bin
ENTRYPOINT ["/usr/local/bin/nvidia_entrypoint.sh"]
