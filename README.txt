To install the package, see this page:

   http://deeplearning.net/software/theano/install.html

For the documentation, see the project website:

   http://deeplearning.net/software/theano/

Related Projects:

   https://github.com/Theano/Theano/wiki/Related-projects

We recommend you look at the documentation on the website, since it
will be more current than the documentation included with the package.
If you really wish to build the documentation yourself, you will need
sphinx. Issue the following command:

    python ./doc/scripts/docgen.py

Documentation is built into html/
The PDF of the documentation is html/theano.pdf


DIRECTORY LAYOUT

Theano (current directory) is the distribution directory.
    * Theano/theano contains the package
    * Theano/theano has several submodules:
        * gof + compile are the core
        * scalar depends upon core
        * tensor depends upon scalar
        * sparse depends upon tensor
        * sandbox can depend on everything else
    * Theano/examples are copies of the example on the wiki
    * Theano/benchmark and Theano/examples are in the distribution, but not in
      the Python package
    * Theano/bin contains executable scripts that are copied to the bin folder
      when the Python package is installed
    * Tests are distributed and are part of the package, i.e. fall in
      the appropriate submodules
    * Theano/doc contains files and scripts used to generate the documentation
    * Theano/html is the place where the documentation will be generated
