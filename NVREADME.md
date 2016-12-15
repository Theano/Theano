Theano
============

## Introduction

Theano is a Python library that allows you to define, optimize, and evaluate
mathematical expressions involving multi-dimensional arrays efficiently. Theano
features:

* Tight integration with NumPy – Use numpy.ndarray in Theano-compiled functions.
* Transparent use of a GPU – Perform data-intensive calculations up to 140x faster than with CPU.(float32 only)
* Efficient symbolic differentiation – Theano does your derivatives for function with one or many inputs.
* Speed and stability optimizations – Get the right answer for log(1+x) even when x is really tiny.
* Dynamic C code generation – Evaluate expressions faster.
* Extensive unit-testing and self-verification – Detect and diagnose many types of errors.

Theano has been powering large-scale computationally intensive scientific
investigations since 2007. But it is also approachable enough to be used in the
classroom.

## Contents

This container has the Theano Python package installed and ready to use.
/opt/theano contains the complete source of this version of Theano.

## Getting Started

The basic command for running containers is to use the ```nvidia-docker run```
command, specifying the URL for the container, which includes the registry
address, repository name, and a version tag, similar to the following:
```$ nvidia-docker run nvcr.io/nvidia/theano:16.12```
There are additional flags and settings that should be used with this command,
as described in the next sections.

### The Remove flag

By default, Docker containers remain on the system after being run.  Repeated
pull/run operations use up more and more space on the local disk, even after
exiting the container.  Therefore, it is important to clean up the Docker
containers after exiting.

To automatically remove a container when exiting, use the ```--rm``` flag:
```$ nvidia-docker run --rm nvcr.io/nvidia/theano:16.12```

### Batch versus Interactive mode

By default, containers run in batch mode; that is, the container is run once
and then exited without any user interaction. Containers can also be run in
interactive mode.

To run in interactive mode, add the ```-ti``` flag to the run command:
```$ nvidia-docker run --rm -ti nvcr.io/nvidia/theano:16.12```

To run in batch mode, leave out the ```-ti``` flag, and instead append the
command to be run in the container to the nvidia-docker run command line:
```$ nvidia-docker run --rm nvcr.io/nvidia/theano:16.12 python myscript.py```

In both cases, it will often be desirable to pull in data and model
descriptions from locations outside the container (e.g., "myscript.py" in the
example above).  To accomplish this, the easiest method is to mount one or more
host directories as [Docker data volumes](https://docs.docker.com/engine/tutorials/dockervolumes/#/mount-a-host-directory-as-a-data-volume)
using the ```-v``` flag of ```nvidia-docker run```.

## Invoking Theano

Theano is run simply by importing it as a Python module:

```
$ python
>>> import numpy
>>> import theano.tensor as T
>>> from theano import function
>>> x = T.dscalar('x')
>>> y = T.dscalar('y')
>>> z = x + y
>>> f = function([x, y], z)
>>> f(2, 3)
array(5.0)
>>> numpy.allclose(f(16.3, 12.1), 28.4)
True
```

## Suggested Reading

A tutorial and documentation are available on the Theano website:

   http://deeplearning.net/software/theano/
   http://deeplearning.net/software/theano/library/
   http://deeplearning.net/software/theano/tutorial/

Related Projects:

   https://github.com/Theano/Theano/wiki/Related-projects

## Customizing the container

If you would like to modify the source code of the version of Theano in this
container and run your customized version, or if you would like to install
additional packages into the container, you can easily use ```docker build``` to
add your customizations on top of this container.

First, create a file called "Dockerfile" describing the modifications you wish
to make -- see https://docs.docker.com/engine/reference/builder/ for a syntax
reference; some examples are below.

Then run a command like the following from the directory containing this new Dockerfile:
```
docker build -t my-custom-theano:version .
```

This will allow you to ```nvidia-docker run ... my-custom-theano:version ...``` in the
same way as you would otherwise have run the stock NVIDIA Theano container.  Further,
it will allow you to "replay" your modifications on top of later NVIDIA Theano containers
simply by updating the NVIDIA version tag in the "FROM" line of your Dockerfile and
rerunning ```docker build```.

### Adding packages
To install additional packages, create a Dockerfile similar to the following:
```
FROM nvcr.io/nvidia/theano:16.12

# Install my-extra-package-1 and my-extra-package-2
RUN apt-get update && apt-get install -y --no-install-recommends \
        my-extra-package-1 \
        my-extra-package-2 \
      && \
    rm -rf /var/lib/apt/lists/
```

### Customizing Theano
To modify and rebuild Theano, create a Dockerfile similar to the following:

```
FROM nvcr.io/nvidia/theano:16.12

# Bring in changes from outside container to /tmp
# (assumes my-theano-modifications.patch is in same directory as Dockerfile)
COPY my-theano-modifications.patch /tmp

# Change working directory to Theano source path
WORKDIR /opt/theano

# Apply modifications
RUN patch -p0 < /tmp/my-theano-modifications.patch

# Rebuild of Theano is unnecessary because it was originally
# installed in pip's editable mode using "pip install -e ."

# Reset default working directory
WORKDIR /workspace
```
