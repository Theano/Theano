"""
This module provides the Scan Op.

Scanning is a general form of recurrence, which can be used for looping.
The idea is that you *scan* a function along some input sequence, producing
an output at each time-step that can be seen (but not modified) by the
function at the next time-step. (Technically, the function can see the
previous K  time-steps of your outputs and L time steps (from the past and
future) of your inputs.

So for example, ``sum()`` could be computed by scanning the ``z+x_i``
function over a list, given an initial state of ``z=0``.

Special cases:

* A *reduce* operation can be performed by returning only the last
  output of a ``scan``.
* A *map* operation can be performed by applying a function that
  ignores previous steps of the outputs.

Often a for-loop can be expressed as a ``scan()`` operation, and ``scan`` is
the closest that theano comes to looping. The advantage of using ``scan``
over for loops is that it allows the number of iterations to be a part of
the symbolic graph.

The Scan Op should typically be used by calling any of the following
functions: ``scan()``, ``map()``, ``reduce()``, ``foldl()``,
``foldr()``.

"""
from __future__ import absolute_import, print_function, division
__docformat__ = 'restructedtext en'
__authors__ = ("Razvan Pascanu "
               "Frederic Bastien "
               "James Bergstra "
               "Pascal Lamblin "
               "Arnaud Bergeron ")
__copyright__ = "(c) 2010, Universite de Montreal"
__contact__ = "Razvan Pascanu <r.pascanu@gmail>"

from theano.scan_module import scan_opt
from theano.scan_module.scan import scan
from theano.scan_module.scan_checkpoints import scan_checkpoints
from theano.scan_module.scan_views import map, reduce, foldl, foldr
from theano.scan_module.scan_utils import clone, until
