============================================================================================================
MILA will stop developing Theano: https://groups.google.com/d/msg/theano-users/7Poq8BZutbY/rNCIfvAEAwAJ

The PyMC developers have forked Theano to a new project called Aesara that is being actively developed: https://github.com/aesara-devs/aesara
============================================================================================================
THEANO:
Theano is a Python library and optimizing compiler for manipulating and evaluating mathematical expressions, especially matrix-valued ones. In Theano, computations are expressed using a NumPy-esque syntax and compiled to run efficiently on either CPU or GPU architectures.Let's learn more!

To install the package, see this page:
   https://theano-pymc.readthedocs.io/en/latest/install.html

For the documentation, see the project website:
   https://theano-pymc.readthedocs.io/en/latest/

Related Projects:
   https://github.com/Theano/Theano/wiki/Related-projects

It is recommended that you look at the documentation on the website, as it will be more current than the documentation included with the package.

In order to build the documentation yourself, you will need sphinx. Issue the following command:

::

   python ./doc/scripts/docgen.py

Documentation is built into ``html/``

The PDF of the documentation can be found at ``html/theano.pdf``

================
DIRECTORY LAYOUT
================

``Theano`` (current directory) is the distribution directory.

* ``Theano/theano`` contains the package
* ``Theano/theano`` has several submodules:
 
  * ``gof`` + ``compile`` are the core
  * ``scalar`` depends upon core
  * ``tensor`` depends upon ``scalar``
  * ``sparse`` depends upon ``tensor``
  * ``sandbox`` can depend on everything else

* ``Theano/examples`` are copies of the example found on the wiki
* ``Theano/benchmark`` and ``Theano/examples`` are in the distribution, but not in
  the Python package
* ``Theano/bin`` contains executable scripts that are copied to the bin folder
  when the Python package is installed
* Tests are distributed and are part of the package, i.e. fall in
  the appropriate submodules
* ``Theano/doc`` contains files and scripts used to generate the documentation
* ``Theano/html`` is where the documentation will be generated
