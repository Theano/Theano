FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

ENV DEBIAN_FRONTEND noninteractive

# Install tools
RUN apt-get update
RUN apt-get install -y build-essential apt-utils wget git dvipng time

# Install magma lib
RUN apt-get install -y libopenblas-dev
RUN wget http://icl.cs.utk.edu/projectsfiles/magma/downloads/magma-2.2.0.tar.gz
RUN tar xvf magma-2.2.0.tar.gz
RUN cp magma-2.2.0/make.inc-examples/make.inc.openblas magma-2.2.0/make.inc
ENV OPENBLASDIR /usr
ENV CUDADIR /usr/local/cuda
RUN (cd magma-2.2.0 && make && make install prefix=/usr/local)
RUN ldconfig

# Setup conda python for Theano
RUN wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
RUN bash Miniconda2-latest-Linux-x86_64.sh -p /miniconda -b
ENV PATH=/miniconda/bin:${PATH}
RUN conda install numpy scipy sympy mkl nose pydot-ng graphviz cython cmake
RUN conda update -y conda
RUN pip install nose-timer parameterized "flake8<3" "sphinx==1.5.2"
RUN pip install --upgrade-strategy only-if-needed git+https://github.com/lebedov/scikit-cuda.git 

# Setup latex for doc test
RUN apt-get install -y texlive-latex-base texlive-latex-extra

# Install SSH server and Java runtime for Jenkins
RUN apt-get install -y openssh-server default-jre-headless

# Add jenkins user and setup environment
RUN useradd -m -s /bin/bash jenkins
RUN echo jenkins:jenkins | chpasswd
RUN echo "export PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/miniconda/bin:\$PATH" >> /home/jenkins/.bashrc
RUN echo "export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:\$LD_LIBRARY_PATH" >> /home/jenkins/.bashrc

# Copy jenkins slave.jar into container
RUN wget -P home/jenkins http://earlgrey.iro.umontreal.ca:8080/jnlpJars/slave.jar
RUN chown -R jenkins:jenkins /home/jenkins/*

# Set launch command as Jenkins slave.jar
CMD java -jar /home/jenkins/slave.jar