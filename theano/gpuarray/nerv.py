# To prevent flake8 error.
from __future__ import print_function, absolute_import, division

raise ImportError(
    "You are importing theano.gpuarray.nerv. "
    "This module was removed as it was based on nervanagpu that is now deprecated. "
    "To still get this module, use Theano 0.9. "
    "More info about nervanagpu here: https://github.com/NervanaSystems/nervanagpu "
    "(viewed on 2017/07/05).")
