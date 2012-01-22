"""
This module provides syntax shortcut for the Scan Op

See scan.py for details on scan
"""

__docformat__ = 'restructedtext en'
__authors__ = ("Razvan Pascanu "
               "Frederic Bastien "
               "James Bergstra "
               "Pascal Lamblin ")
__copyright__ = "(c) 2010, Universite de Montreal"
__contact__ = "Razvan Pascanu <r.pascanu@gmail>"


import logging

import scan

# Logging function for sending warning or info
_logger = logging.getLogger('theano.scan_module.scan_views')


################ Declaration of Views for Scan #######################


# The ``map`` view of Scan Op.


def map(fn,
        sequences,
        non_sequences=None,
        truncate_gradient=-1,
        go_backwards=False,
        mode=None,
        name=None):
    """
    Similar behaviour as python's map.

    :param fn: The function that ``map`` applies at each iteration step
               (see ``scan`` for more info).

    :param sequences: List of sequences over which ``map`` iterates
                      (see ``scan`` for more info).

    :param non_sequences: List of arguments passed to ``fn``. ``map`` will
                          not iterate over these arguments (see ``scan`` for
                          more info).

    :param truncate_gradient: See ``scan``.

    :param go_backwards: Boolean value that decides the direction of
                         iteration. True means that sequences are parsed
                         from the end towards the begining, while False
                         is the other way around.

    :param mode: See ``scan``.

    :param name: See ``scan``.
    """
    return scan.scan(fn=fn,
                     sequences=sequences,
                     outputs_info=[],
                     non_sequences=non_sequences,
                     truncate_gradient=truncate_gradient,
                     go_backwards=go_backwards,
                     mode=mode,
                     name=name)


# The ``reduce`` view of Scan Op.
def reduce(fn,
           sequences,
           outputs_info,
           non_sequences=None,
           go_backwards=False,
           mode=None,
           name=None):
    """
    Similar behaviour as python's reduce

    :param fn: The function that ``reduce`` applies at each iteration step
               (see ``scan``  for more info).

    :param sequences: List of sequences over which ``reduce`` iterates
                      (see ``scan`` for more info)

    :param outputs_info: List of dictionaries describing the outputs of
                        reduce (see ``scan`` for more info).

    :param non_sequences: List of arguments passed to ``fn``. ``reduce`` will
                          not iterate over these arguments (see ``scan`` for
                          more info).

    :param go_backwards: Boolean value that decides the direction of
                         iteration. True means that sequences are parsed
                         from the end towards the begining, while False
                         is the other way around.

    :param mode: See ``scan``.

    :param name: See ``scan``.
    """
    rval = scan.scan(fn=fn,
                     sequences=sequences,
                     outputs_info=outputs_info,
                     non_sequences=non_sequences,
                     go_backwards=go_backwards,
                     truncate_gradient=-1,
                     mode=mode,
                     name=name)
    if isinstance(rval[0], (list, tuple)):
        return [x[-1] for x in rval[0]], rval[1]
    else:
        return rval[0][-1], rval[1]


# The ``foldl`` view of Scan Op.
def foldl(fn,
          sequences,
          outputs_info,
          non_sequences=None,
          mode=None,
          name=None):
    """
    Similar behaviour as haskell's foldl

    :param fn: The function that ``foldl`` applies at each iteration step
               (see ``scan`` for more info).


    :param sequences: List of sequences over which ``foldl`` iterates
                      (see ``scan`` for more info)

    :param outputs_info: List of dictionaries describing the outputs of
                        reduce (see ``scan`` for more info).

    :param non_sequences: List of arguments passed to `fn`. ``foldl`` will
                          not iterate over these arguments (see ``scan`` for
                          more info).

    :param mode: See ``scan``.

    :param name: See ``scan``.
    """
    return reduce(fn=fn,
                  sequences=sequences,
                  outputs_info=outputs_info,
                  non_sequences=non_sequences,
                  go_backwards=False,
                  mode=mode,
                  name=name)


# The ``foldl`` view of Scan Op.
def foldr(fn,
          sequences,
          outputs_info,
          non_sequences=None,
          mode=None,
          name=None):
    """
    Similar behaviour as haskell' foldr

    :param fn: The function that ``foldr`` applies at each iteration step
               (see ``scan`` for more info).


    :param sequences: List of sequences over which ``foldr`` iterates
                      (see ``scan`` for more info)

    :param outputs_info: List of dictionaries describing the outputs of
                        reduce (see ``scan`` for more info).

    :param non_sequences: List of arguments passed to `fn`. ``foldr`` will
                          not iterate over these arguments (see ``scan`` for
                          more info).

    :param mode: See ``scan``.

    :param name: See ``scan``.
    """
    return reduce(fn=fn,
                  sequences=sequences,
                  outputs_info=outputs_info,
                  non_sequences=non_sequences,
                  go_backwards=True,
                  mode=mode,
                  name=name)
