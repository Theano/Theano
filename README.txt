DIRECTORY LAYOUT
Theano (current directory) is the distribution directory.
    * Theano/theano contains the package
    * Theano/theano has several submodules:
        * gof + compile are the core
        * scalar depends upon core
        * tensor depends upon scalar
        * sparse depends upon tensor
        * sandbox can depends on everything else
    * Theano/examples are copies of the example on the wiki
    * Theano/benchmark, Theano/bin and Theano/examples are in the distribution,
      but not in the python package
    * Tests are distributed and are part of the package, i.e. fall in
      the appropriate submodules
    * Theano/doc contains files and scripts used to generate the documentation
    * Theano/html is the place where the documentation will be generated
