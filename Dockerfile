FROM nvdl.githost.io:4678/dgx/cuda:8.0-cudnn5-devel-ubuntu14.04
MAINTAINER NVIDIA CORPORATION <cudatools@nvidia.com>

ENV THEANO_VERSION 0.8
LABEL com.nvidia.theano.version="0.8"

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libopenblas-dev \
        python-dev \
        python-pip \
        python-nose \
        python-numpy \
        python-scipy && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade nose nose-parameterized

WORKDIR /workspace
COPY . .

RUN python setup.py develop

COPY theanorc /workspace/.theanorc
ENV THEANORC /workspace/.theanorc
