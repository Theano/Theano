# This script allows to run one specific cuDNN convolution test case.
# This script should not be imported, but only used as a program.
# python run_dnn_conv.py --help         # Print help.
# python run_dnn_conv.py {fwd|bwd-filter|bwd-data} {2d|3d} -a <algo> -i <inputShape> -f <filterShape> ...

from __future__ import absolute_import, print_function, division

import argparse
import sys

import theano
from theano.configdefaults import SUPPORTED_DNN_CONV_ALGO_RUNTIME
from theano.gpuarray.cudnn_defs import (HALF, FLOAT, DOUBLE,
                                        TRUE_HALF_CONFIG, PSEUDO_HALF_CONFIG, FLOAT_CONFIG, DOUBLE_CONFIG)
from theano.gpuarray.tests.check_dnn_conv import (cudnn, TestDnnConv2D, TestDnnConv3D, CheckDnn)
from theano.tensor.nnet.abstract_conv import get_conv_output_shape

if __name__ != '__main__':
    raise ImportError('This script must not be imported.')


class TupleAction(argparse.Action):
    # Tuple extractor for command line args parser.
    def __call__(self, parser, namespace, values, option_string=None):
        values = tuple(int(v) for v in values.split(','))
        setattr(namespace, self.dest, values)


class BorderAction(TupleAction):
    # Border extractor for command line args parser.
    def __call__(self, parser, namespace, values, option_string=None):
        if values not in ('valid', 'full', 'half'):
            super(BorderAction, self).__call__(parser, namespace, values, option_string)
        else:
            setattr(namespace, self.dest, values)


args = sys.argv[1:]
computations = FWD, BWD_FILTER, BWD_DATA = ('fwd', 'gradweight', 'gradinput')
algorithms = (tuple(sorted(list(set(cudnn.cudnnConvolutionFwdAlgo_t.get_aliases() +
                                    cudnn.cudnnConvolutionBwdFilterAlgo_t.get_aliases() +
                                    cudnn.cudnnConvolutionBwdDataAlgo_t.get_aliases())))) +
              SUPPORTED_DNN_CONV_ALGO_RUNTIME)
types = (HALF, FLOAT, DOUBLE)
data_type_configurations = dict(TRUE_HALF_CONFIG=TRUE_HALF_CONFIG, PSEUDO_HALF_CONFIG=PSEUDO_HALF_CONFIG,
                                FLOAT_CONFIG=FLOAT_CONFIG, DOUBLE_CONFIG=DOUBLE_CONFIG)

parser = argparse.ArgumentParser()

parser.add_argument('computation', choices=computations,
                    help='Computation to run.')

parser.add_argument('-a', '--algo', choices=algorithms, required=True,
                    help='Algorithm to use for computation.')
parser.add_argument('-i', '--input-shape', action=TupleAction, required=True,
                    help='Input shape. Comma-separated list of integers (no spaces).')
parser.add_argument('-f', '--filter-shape', action=TupleAction, required=True,
                    help='Filter shape. Comma-separated list of integers (no spaces).')

parser.add_argument('-D', '--dtype-config', choices=list(sorted(data_type_configurations.keys())), default=None,
                    help='Data type configuration for (data type; precision). Default (theano floatX; theano floatX). '
                         'To specify data type configuration, you can either use this option or set data type and '
                         'precision separately with "-t" and "-p" options.')
parser.add_argument('-t', '--dtype', choices=types, default=None,
                    help='Data type (default theano floatX).')
parser.add_argument('-p', '--precision', choices=types, default=None,
                    help='Precision (default theano floatX).')
parser.add_argument('-s', '--subsample', action=TupleAction,
                    help='Subsample. Comma-separated list of integers (no spaces). '
                         'Default: 1 per dimension.')
parser.add_argument('-d', '--dilation', action=TupleAction,
                    help='Dilation. Comma-separated list of integers (no spaces). '
                         'Default: 1 per dimension.')
parser.add_argument('-b', '--border-mode', default='valid', action=BorderAction,
                    help='Border mode. "valid" (default), "full", "half" '
                         'or a comma-separated list of integers (no spaces).')
parser.add_argument('-c', '--conv-mode', choices=('conv', 'cross'), default='conv',
                    help='Conv mode (default: conv).')
parser.add_argument('-A', '--alpha', type=float, default=1,
                    help="alpha (floating), must not be zero. Default 1.")
parser.add_argument('-B', '--beta', type=float, default=0,
                    help='beta (floating). Default 0.')

parser.add_argument('-I', '--print-infos', action='store_true', default=False,
                    help='Print some infos before testing.')

args = parser.parse_args(args)

test = args.computation
if len(args.input_shape) != len(args.filter_shape):
    raise ValueError('Expected same length for input shape and filter shape')
if len(args.input_shape) not in (4, 5):
    raise ValueError('Expected length 4 or 5 for input shape')
ndim = len(args.input_shape) - 2
if ndim == 2:
    tests = TestDnnConv2D()
elif ndim == 3:
    tests = TestDnnConv3D()

if args.subsample is None:
    args.subsample = (1,) * ndim
if args.dilation is None:
    args.dilation = (1,) * ndim
if not (ndim == len(args.subsample) == len(args.dilation)):
    raise ValueError('Expected parameters sized for %d dimensions.' % ndim)

if isinstance(args.border_mode, tuple) and ndim != len(args.border_mode):
    raise ValueError('Expected borders sized for %d dimensions.' % ndim)

if args.alpha == 0:
    raise ValueError('Nothing could be computed if alpha is 0.')

if args.dtype_config is None:
    if args.dtype is None:
        args.dtype = theano.config.floatX
    if args.precision is None:
        args.precision = theano.config.floatX
else:
    if args.dtype is not None or args.precision is not None:
        raise ValueError('You must specify either -D <data-type-configuration> '
                         'or (-t <data-type> -p <precision>), not both.')
    args.dtype, args.precision = data_type_configurations[args.dtype_config]
if (args.dtype, args.precision) not in cudnn.get_supported_dtype_configs():
    raise ValueError('Unsupported data type configuration %s %s.' % (args.dtype, args.precision))

if args.algo not in SUPPORTED_DNN_CONV_ALGO_RUNTIME:
    check_config = False
    if test == FWD:
        check_config = cudnn.fwd_algo_supports_dtype_config(args.algo, args.dtype, args.precision, ndim)
    elif test == BWD_FILTER:
        check_config = cudnn.bwd_filter_algo_supports_dtype_config(args.algo, args.dtype, args.precision, ndim)
    elif test == BWD_DATA:
        check_config = cudnn.bwd_data_algo_supports_dtype_config(args.algo, args.dtype, args.precision, ndim)
    if not check_config:
        print('Warning: %s computation does not normally support configuration (%s, %s) for algo %s.' % (
            test, args.dtype, args.precision, args.algo), file=sys.stderr)

algo = args.algo
dtype = args.dtype
precision = args.precision
parameters = (
    args.input_shape, args.filter_shape, args.subsample, args.dilation, args.border_mode, args.conv_mode,
    args.alpha, args.beta)
if args.print_infos:
    CheckDnn.print_infos(count_tests=False)
print('======================')
print('Running', test, algo, dtype, precision, *parameters)
if test == FWD:
    tests.run_conv_fwd(algo, dtype, precision, parameters)
    expected_output_shape = get_conv_output_shape(args.input_shape, args.filter_shape, args.border_mode,
                                                  args.subsample, args.dilation)
elif test == BWD_FILTER:
    tests.run_conv_gradweight(algo, dtype, precision, parameters)
    expected_output_shape = args.filter_shape
elif test == BWD_DATA:
    tests.run_conv_gradinput(algo, dtype, precision, parameters)
    expected_output_shape = args.input_shape
print('Computed shape:', expected_output_shape)
print('... OK')
