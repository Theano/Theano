FROM nvdl.githost.io:4678/dgx/cuda:8.0-cudnn5-devel-ubuntu14.04
MAINTAINER NVIDIA CORPORATION <cudatools@nvidia.com>

ENV THEANO_VERSION 0.8.2
LABEL com.nvidia.theano.version="0.8.2"

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

RUN umask 0000 & \
    pip install -e .

WORKDIR /workspace
COPY README.txt .
COPY LICENSE.txt .
COPY theanorc /workspace/.theanorc
ENV THEANORC /workspace/.theanorc

RUN ln -sf /opt/theano/benchmark /workspace && \
    chmod a+w /opt/theano /workspace
