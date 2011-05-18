"""
This module provides syntax shortcut for the Scan Op

See scan.py for details on scan
"""

__docformat__ = 'restructedtext en'
__authors__ = ( "Razvan Pascanu "
                "Frederic Bastien "
                "James Bergstra "
                "Pascal Lamblin "  )
__copyright__ = "(c) 2010, Universite de Montreal"
__contact__ = "Razvan Pascanu <r.pascanu@gmail>"


import logging

import scan

# Logging function for sending warning or info
_logger = logging.getLogger('theano.scan_views')

def warning(*msg):
    _logger.warning('WARNING theano.scan: '+' '.join(msg))

def info(*msg):
    _logger.info('INFO theano.scan: '+' '.join(msg))


################ Declaration of Views for Scan #######################


# The ``map`` view of Scan Op.


def map( fn
        , sequences
        , non_sequences     = None
        , truncate_gradient = -1
        , go_backwards      = False
        , mode              = None
        , name              = None  ):
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
    return scan.scan( fn                 = fn
                , sequences         = sequences
                , outputs_info      = []
                , non_sequences     = non_sequences
                , truncate_gradient = truncate_gradient
                , go_backwards      = go_backwards
                , mode              = mode
                , name              = name )


# The ``reduce`` view of Scan Op.
def reduce( fn
           , sequences
           , outputs_info
           , non_sequences = None
           , go_backwards  = False
           , mode          = None
           , name          = None ):
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
    # Makes sure the outputs_info is a list.
    if not isinstance(outputs_info, (list,tuple)):
        outs_info = [outputs_info]
    else:
        outs_info = list(outputs_info)

    for i,out_info in enumerate(outs_info):
        if out_info:
            if not isinstance(out_info, dict):
                # Specifies that it should return only the last step.
                outs_info[i] = dict(
                    initial = out_info,  return_steps = 1)
            else:
                # Specifies that it should return only the last step.
                outs_info[i]['return_steps'] = 1
                # NOTE : If the user asks for more then the last step,
                # it means he does not understand ``reduce``. We could
                # issue a warning in that case
    return scan.scan( fn                 = fn
                , sequences         = sequences
                , outputs_info      = outs_info
                , non_sequences     = non_sequences
                , go_backwards      = go_backwards
                , truncate_gradient = -1
                , mode              = mode
                , name              = name )


# The ``foldl`` view of Scan Op.
def foldl( fn
          , sequences
          , outputs_info
          , non_sequences = None
          , mode          = None
          , name          = None  ):
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
    return reduce( fn             = fn
                  , sequences     = sequences
                  , outputs_info  = outputs_info
                  , non_sequences = non_sequences
                  , go_backwards  = False
                  , mode          = mode
                  , name          = name )


# The ``foldl`` view of Scan Op.
def foldr( fn
          , sequences
          , outputs_info
          , non_sequences = None
          , mode          = None
          , name          = None ):
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
    return reduce( fn             = fn
                  , sequences     = sequences
                  , outputs_info  = outputs_info
                  , non_sequences = non_sequences
                  , go_backwards  = True
                  , mode          = mode
                  , name          = name )






