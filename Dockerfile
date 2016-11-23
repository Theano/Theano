FROM nvdl.githost.io:4678/dgx/cuda:8.0-cudnn5-devel-ubuntu14.04
MAINTAINER NVIDIA CORPORATION <cudatools@nvidia.com>

ENV THEANO_VERSION 0.8.2
LABEL com.nvidia.theano.version="0.8.2"

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libopenblas-dev \
        python-dev \
        python-pip \
        python-nose \
        python-numpy \
        python-scipy && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel
RUN pip install --upgrade --no-cache-dir nose nose-parameterized

WORKDIR /workspace
COPY . .

RUN pip install -e .

COPY theanorc /workspace/.theanorc
ENV THEANORC /workspace/.theanorc

RUN chmod -R a+w /workspace

################################################################################
# Show installed packages
################################################################################

RUN echo "------------------------------------------------------" && \
    echo "-- INSTALLED PACKAGES --------------------------------" && \
    echo "------------------------------------------------------" && \
    echo "[[dpkg -l]]" && \
    dpkg -l && \
    echo "" && \
    echo "[[pip list]]" && \
    pip list && \
    echo "" && \
    echo "------------------------------------------------------" && \
    echo "-- FILE SIZE, DATE, HASH -----------------------------" && \
    echo "------------------------------------------------------" && \
    echo "[[find /usr/bin /usr/sbin /usr/lib /usr/local /workspace -type f | xargs ls -al]]" && \
    (find /usr/bin /usr/sbin /usr/lib /usr/local /workspace -type f | xargs ls -al || true) && \
    echo "" && \
    echo "[[find /usr/bin /usr/sbin /usr/lib /usr/local /workspace -type f | xargs md5sum]]" && \
    (find /usr/bin /usr/sbin /usr/lib /usr/local /workspace -type f | xargs md5sum || true)
