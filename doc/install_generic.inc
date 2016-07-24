.. include:: css.inc

.. _install_generic:


Installation
============

Stable Installation
-------------------

Install the latest stable version of Theano with:

.. raw:: html

    <div class="highlight"><pre><span class="red">&lt;sudo&gt;</span> pip install <span class="blue">&lt;--user&gt;</span> Theano[test, doc]</pre></div>

- Any argument between <...> is optional.

- Use :red:`sudo` for a root installation.

- Use :blue:`user` for a user installation without admin rights. It will install Theano in your local site-packages.

- [test] will install the requirements for testing.

- [doc] will install the requirements in order to generate the documentation.

If you encountered any trouble, head to the :ref:`troubleshooting` page.

libgpuarray
^^^^^^^^^^^

For the stable version of Theano you need a specific version of libgpuarray, 
that has been tagged ``v-9998``.
Download it with:

.. raw:: html

    <div class='highlight'><pre>
    git clone https://github.com/Theano/libgpuarray.git --tags
    git checkout origin/v-9998
    cd libgpuarray
    </pre></div>

and then follow the `Step-by-step instructions <http://deeplearning.net/software/libgpuarray/installation.html#step-by-step-install>`__.


Bleeding-Edge Installation (recommended)
----------------------------------------

Install the latest, bleeding-edge, development version of Theano with:

.. raw:: html

    <div class='highlight'><pre><span class="red">&lt;sudo&gt;</span> pip install <span class="blue">&lt;--user&gt;</span> <span class="pink">&lt;--no-deps&gt;</span> git+https://github.com/Theano/Theano.git#egg=Theano</pre></div>

- Any argument between <...> is optional.

- Use :red:`sudo` for a root installation.

- Use :blue:`user` for a user installation without admin rights. It will install Theano in your local site-packages.

- Use :pink:`no-deps` when you don't want the dependencies of Theano to be installed through pip. This is important when they have already been installed as system packages.

If you encountered any trouble, head to the :ref:`troubleshooting` page.

libgpuarray
^^^^^^^^^^^

Install the latest, development version of libgpuarray following the
`Step-by-step instructions <http://deeplearning.net/software/libgpuarray/installation.html#step-by-step-install>`__.


Developer Installation
----------------------

Install the developer version of Theano with:

.. raw:: html

    <div class="highlight"><pre>git clone git://github.com/Theano/Theano.git
    cd Theano
    <span class="red">&lt;sudo&gt;</span> pip install <span class="blue">&lt;--user&gt;</span> <span class="pink">&lt;--no-deps&gt;</span> <span class="green">-e .</span></pre></div>

- Any argument between <...> is optional.

- Use :red:`sudo` for a root installation.

- Use :blue:`user` for a user installation without admin rights. It will install Theano in your local site-packages.

- Use :pink:`no-deps` when you don't want the dependencies of Theano to be installed through pip. This is important when they have already been installed as system packages.

- :green:`-e` makes your installation *editable*, i.e., it links it to your
  source directory.

If you encountered any trouble, head to the :ref:`troubleshooting` page.

libgpuarray
^^^^^^^^^^^

Install the latest, development version of libgpuarray following the
`Step-by-step instructions <http://deeplearning.net/software/libgpuarray/installation.html#step-by-step-install>`__.
