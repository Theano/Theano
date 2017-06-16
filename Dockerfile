FROM nvdl.githost.io:4678/dgx/cuda:8.0-cudnn6-devel-ubuntu16.04--17.07

ENV THEANO_VERSION 0.9.0
LABEL com.nvidia.theano.version="${THEANO_VERSION}"
ENV NVIDIA_THEANO_VERSION 17.07

RUN apt-get update && apt-get install -y --upgrade --no-install-recommends \
        cmake \
        exuberant-ctags \
        gfortran \
        graphviz \
        libopenblas-dev \
	apt-utils dvipng time curl \
        python-dev && \
    rm -rf /var/lib/apt/lists/* 

WORKDIR /tmp
RUN curl -O http://icl.cs.utk.edu/projectsfiles/magma/downloads/magma-2.2.0.tar.gz
RUN tar xvf magma-2.2.0.tar.gz
RUN cp magma-2.2.0/make.inc-examples/make.inc.openblas magma-2.2.0/make.inc
ENV OPENBLASDIR /usr
ENV CUDADIR /usr/local/cuda
RUN (cd magma-2.2.0 && make GPU_TARGET="sm50 sm52 sm60 sm61" && make install prefix=/usr/local)
RUN ldconfig

WORKDIR /tmp
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN pip install --upgrade --no-cache-dir pip setuptools wheel

ARG NVIDIA_BUILD_ID
ENV NVIDIA_BUILD_ID ${NVIDIA_BUILD_ID:-<unknown>}
LABEL com.nvidia.build.id="${NVIDIA_BUILD_ID}"
ARG NVIDIA_BUILD_REF
LABEL com.nvidia.build.ref="${NVIDIA_BUILD_REF}"

WORKDIR /opt/theano
COPY . .

RUN patch -p0 < bug1893551.patch && rm bug1893551.patch

RUN MAKEFLAGS="-j$(nproc)" \
    PREFIX=/usr/local \
    ./install.sh 

WORKDIR /workspace
COPY NVREADME.md README.md
COPY docker-examples docker-examples
COPY theanorc /workspace/.theanorc
ENV THEANORC /workspace/.theanorc
RUN chmod -R a+w /workspace

COPY nvidia_entrypoint.sh /usr/local/bin
ENTRYPOINT ["/usr/local/bin/nvidia_entrypoint.sh"]
