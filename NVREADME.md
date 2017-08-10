Theano
======

Theano is a Python library that allows you to define, optimize, and evaluate
mathematical expressions involving multi-dimensional arrays efficiently. Theano
features:

* Tight integration with NumPy -- Use numpy.ndarray in Theano-compiled functions.
* Transparent use of a GPU -- Perform data-intensive calculations up to 140x faster than with CPU.(float32 only)
* Efficient symbolic differentiation -- Theano does your derivatives for function with one or many inputs.
* Speed and stability optimizations -- Get the right answer for log(1+x) even when x is really tiny.
* Dynamic C code generation -- Evaluate expressions faster.
* Extensive unit-testing and self-verification -- Detect and diagnose many types of errors.

Theano has been powering large-scale computationally intensive scientific
investigations since 2007. But it is also approachable enough to be used in the
classroom.

## Contents

This container has the Theano Python package installed and ready to use.
`/opt/theano` contains the complete source of this version of Theano.

## Running Theano

You can choose to use Theano as provided by NVIDIA, or you can choose to
customize it.

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

## Customizing Theano

You can customize Theano one of two ways:

(1) Modify the version of the source code in this container and run your
customized version, or (2) use `docker build` to add your customizations on top
of this container if you want to add additional packages.

NVIDIA recommends option 2 for ease of migration to later versions of the
Theano container image.

For more information, see https://docs.docker.com/engine/reference/builder/ for
a syntax reference.  Several example Dockerfiles are provided in the container
image in `/workspace/docker-examples`.

## Suggested Reading

For more information about Theano, including tutorials, documentation, and
examples, see the [Theano webpage](http://deeplearning.net/software/theano)
and the [Theano project](https://github.com/Theano/Theano/wiki/Related-projects).
