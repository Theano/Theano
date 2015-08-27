"""
This module contains the configuration flags for cudnn support.

Those are shared between the cuda and gpuarray backend which is why
they are in this file.
"""
import os.path

from theano.configparser import AddConfigVar, EnumStr, StrParam
from theano import config

AddConfigVar('dnn.conv.workmem',
             "This flag is deprecated; use dnn.conv.algo_fwd.",
             EnumStr(''),
             in_c_key=False)

AddConfigVar('dnn.conv.workmem_bwd',
             "This flag is deprecated; use dnn.conv.algo_bwd.",
             EnumStr(''),
             in_c_key=False)

AddConfigVar('dnn.conv.algo_fwd',
             "Default implementation to use for CuDNN forward convolution.",
             EnumStr('small', 'none', 'large', 'fft', 'guess_once',
                     'guess_on_shape_change', 'time_once',
                     'time_on_shape_change'),
             in_c_key=False)

AddConfigVar('dnn.conv.algo_bwd',
             "Default implementation to use for CuDNN backward convolution.",
             EnumStr('none', 'deterministic', 'fft', 'guess_once',
                     'guess_on_shape_change', 'time_once',
                     'time_on_shape_change'),
             in_c_key=False)

AddConfigVar('dnn.include_path',
             "Location of the cudnn header (defaults to the cuda root)",
             StrParam(lambda: os.path.join(config.cuda.root, 'include')))

AddConfigVar('dnn.library_path',
             "Location of the cudnn header (defaults to the cuda root)",
             StrParam(lambda: os.path.join(config.cuda.root, 'lib64')))
