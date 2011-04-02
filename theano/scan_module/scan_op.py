"""
This module provides the Scan Op

See scan.py for details on scan
"""

__docformat__ = 'restructedtext en'
__authors__ = ( "Razvan Pascanu "
                "Frederic Bastien "
                "James Bergstra "
                "Pascal Lamblin "  )
__copyright__ = "(c) 2010, Universite de Montreal"
__contact__ = "Razvan Pascanu <r.pascanu@gmail>"

import copy
import itertools
import logging
import numpy

from theano.compile import SharedVariable, function, Param
from theano import compile
from theano import gradient
from theano.gof.python25 import all
from theano.gof import Op, Apply
from theano import gof
from theano.misc import safe_asarray as safe_asarray
from theano.tensor import TensorType
from theano import tensor
from theano.tensor.opt import Shape_i
import theano

import scan_utils
from scan_utils import safe_new, safe_to_cpu, traverse

# Logging function for sending warning or info
_logger = logging.getLogger('theano.scan_op')


def warning(*msg):
    _logger.warning('WARNING theano.scan: '+' '.join(msg))


def info(*msg):
    _logger.info('INFO theano.scan: '+' '.join(msg))

from theano.sandbox import cuda

class Scan(Op):
    #
    # OLD DOCUMENTATION CAN BE FOUND NEAR REVISION 2581
    #

    def __init__( self
                 , inputs
                 , outputs
                 , info  ):
        """
        :param inputs: inputs of the inner function of scan
        :param outputs: outputs of the inner function of scan
        :param properties: dictionary containing different properties of
                        the scan op.
        """
        # adding properties into self
        self.info    = info
        self.inputs  = inputs
        self.outputs = outputs
        self.__dict__.update(info)

        # build a list of output types for any Apply node using this op.
        info['output_types'] = []
        idx = 0
        jdx = 0
        if info['gpu']:
            # mit_mot
            while idx < self.n_mit_mot_outs:
                # Not that for mit_mot there are several output slices per
                # output sequence
                o     = outputs[idx]
                info['output_types'].append(
                    cuda.CudaNdarrayType(
                        broadcastable = (False,) + o.type.broadcastable))
                idx += len(self.mit_mot_out_slices[jdx])
                jdx += 1

            # mit_sot / sit_sot / nit_sot
            end = idx + self.n_mit_sot + self.n_sit_sot + self.n_nit_sot
            for o in outputs[idx:end]:
                info['output_types'].append(
                    cuda.CudaNdarrayType( broadcastable = (False,) +
                                    o.type.broadcastable))
            # shared outputs
            for o in outputs[end:]:
                if isinstance(o.type, TensorType):
                    info['output_types'].append(cuda.CudaNdarrayType(
                        broadcastable = o.type.broadcastable))
                else:
                    info['output_types'].append( o.type )
        else:
            while idx < self.n_mit_mot_outs:
                # Not that for mit_mot there are several output slices per
                # output sequence
                o     = outputs[idx]
                info['output_types'].append(
                    TensorType(
                        broadcastable = (False,) + o.type.broadcastable
                        , dtype = o.type.dtype)
                    )
                idx += len(self.mit_mot_out_slices[jdx])
                jdx += 1

            # mit_sot / sit_sot / nit_sot
            end = idx + self.n_mit_sot + self.n_sit_sot + self.n_nit_sot
            for o in outputs[idx:end]:
                info['output_types'].append(
                    TensorType(
                        broadcastable = (False,) + o.type.broadcastable
                        , dtype = o.type.dtype ))
            # shared outputs
            for o in outputs[end:]:
                if cuda.cuda_available and isinstance(o.type,
                                                      cuda.CudaNdarrayType):
                    info['output_types'].append( TensorType(
                        broadcastable = o.type.broadcastable
                        , dtype = theano.config.floatX) )
                else:
                    info['output_types'].append( o.type )


        self.destroy_map = {}

        if 'inplace' in info and info['inplace']:
            for idx in xrange(info['n_mit_mot'] + info['n_mit_sot'] +
                              info['n_sit_sot'] ):
                self.destroy_map[idx] = [idx + 1 + info['n_seqs']]

        # I consider all inputs of the inner function non mutable
        nonmutable = range(len(inputs))

        mode_instance = compile.mode.get_mode(info['mode'])
        # if the default mode is used, and that mode is ProfileMode
        # then we need to copy the mode otherwise the time for a given
        # op will be counted multiple times
        if ( info['mode'] is None and
            isinstance(mode_instance, compile.profilemode.ProfileMode) ):
            mode_instance = compile.profilemode.ProfileMode(
                optimizer = mode_instance.provided_optimizer
                , linker = mode_instance.provided_linker )
            compile.profilemode.prof_mode_instance_to_print.append(mode_instance)
            info['mode_instance'] = mode_instance
            if self.name:
                info['mode_instance'].message = self.name + " sub profile"
            else:
                info['mode_instance'].message = "Scan sub profile"
        else:
            info['mode_instance'] = mode_instance

        if 'name' not in info or info['name'] is None:
            info['name'] = 'scan_fn'

        if isinstance(info['mode_instance'], compile.debugmode.DebugMode):
            theano_fn = function(
                inputs
                , outputs
                , mode = info['mode_instance']
                , name = info['name'] )

            def fn_wrapper(ins_storage, outs_storage):
                '''
                 Wrap theano_fn to have same interface as scan_utils's
                 scan_function
                '''
                outputs = theano_fn(*ins_storage)
                for (out,out_storage) in zip( outputs, outs_storage):
                    if out_storage[0] is not None and out_storage[0].shape:
                        out_storage[0][:] = out
                    elif out_storage[0] is not None:
                        out_storage[0].itemset(out)
                return [[o] for o in outputs ]
            self.fn               = fn_wrapper
            self.fn.maker         = scan_utils.EmptyObject()
            self.fn.maker.inputs  = inputs
            self.fn.maker.outputs = outputs
            self.fn.maker.env     = theano_fn.maker.env
            self.mask = [ 0 for x in xrange(self.n_shared_outs)]
        else:
            self.mask, self.fn = scan_utils.scan_function(
                            inputs
                            , outputs
                            , nonmutable
                            , mode = info['mode_instance']
                            , name = info['name']
                            , slices = ( info['n_mit_mot_outs'] +
                                         info['n_mit_sot'] +
                                         info['n_sit_sot'] +
                                         info['n_nit_sot'] )

                            )
            # check for shared variables in the inputs
            assert not numpy.any( [isinstance(x, SharedVariable) for x
                           in self.fn.maker.inputs])

        self.__dict__.update(info)
        self.info = info
        # Pre-computing some values to speed up perform
        self.mintaps   = [ numpy.min(x) for x in self.tap_array]
        self.mintaps  += [ 0 for x in xrange(self.n_nit_sot) ]
        self.seqs_arg_offset = 1+self.n_seqs
        self.shared_arg_offset = ( self.seqs_arg_offset
                                + self.n_mit_mot
                                + self.n_mit_sot
                                + self.n_sit_sot )
        self.nit_sot_arg_offset = ( self.shared_arg_offset +
                                    self.n_shared_outs )
        self.n_outs = self.n_mit_mot + self.n_mit_sot + self.n_sit_sot
        self.n_tap_outs = self.n_mit_mot + self.n_mit_sot

    def make_node(self, *inputs):
        assert numpy.all(isinstance(i, gof.Variable) for i in inputs)
        # assert dtype is consistent
        err_msg1 = ('%s %s (index %d) has dtype %s. Slice %s representing '
                   'this input has dtype %s' )

        err_msg2 = ('Initial state %s (index %d) has dtype %s. The '
                    'corresponding output of the inner function applied '
                    'recurrently has dtype %s')

        # Flags that indicate which inputs are vectors

        self.vector_seqs = [ seq.ndim == 1 for seq in
                             inputs[1:1+self.n_seqs ] ]
        self.vector_outs = [ arg.ndim ==1 for arg in
                             inputs[1+self.n_seqs: (1+self.n_seqs +
                                                    self.n_outs)] ]
        self.vector_outs += [ False]*self.n_nit_sot

        # Check if input sequences and variables representing a slice of
        # them have the same dtype
        for idx in xrange(self.n_seqs):
            if inputs[1+idx].dtype != self.inputs[idx].dtype:
                raise ValueError(err_msg1%( 'Sequence'
                                       , inputs[1+idx].name
                                       , idx
                                       , inputs[1+idx].dtype
                                       , self.inputs[idx].name
                                       , self.inputs[idx].dtype) )

        # Check that this 3 things have the same dtype for mit_mot:
        #   - initial state of the output
        #   - variable representing an input slice of the otuput
        #   - variable representing an output slice of the otuput
        # Maybe checking that ndim fits would be good as well !?
        index_i = self.n_seqs
        index_o = 0
        index   = 1+self.n_seqs
        start   = index
        end     = index + self.n_mit_mot
        while index < end:
            for k in self.tap_array[index-start]:
                if inputs[index].dtype != self.inputs[index_i].dtype:
                    raise ValueError(err_msg1%( 'Initial state'
                                               , inputs[index].name
                                               , idx
                                               , inputs[index].dtype
                                               , self.inputs[index_i].name
                                               , self.inputs[index_i].dtype) )
                index_i += 1
            for k in self.mit_mot_out_slices[index-start]:
                if inputs[index].dtype != self.outputs[index_o].dtype:
                    raise ValueError(err_msg2%( inputs[index].name
                                               , idx
                                               , inputs[index].dtype
                                               , self.outputs[index_o].dtype) )
                index_o += 1
            index += 1
        # Same checks as above but for outputs of type mit_sot and sit_sot
        end += self.n_mit_sot + self.n_sit_sot
        while index < end:
            for k in self.tap_array[index-start]:
                if inputs[index].dtype != self.inputs[index_i].dtype:
                    raise ValueError(err_msg1%( 'Initial state'
                                               , inputs[index].name
                                               , idx
                                               , inputs[index].dtype
                                               , self.inputs[index_i].name
                                               , self.inputs[index_i].dtype) )
                index_i += 1
            if inputs[index].dtype != self.outputs[index_o].dtype:
                raise ValueError(err_msg2%( inputs[index].name
                                           , index
                                           , inputs[index].dtype
                                           , self.outputs[index_o].dtype) )
            index_o += 1
            index   += 1

        # Check that the shared variable and their update rule have the same
        # dtype. Maybe even same type ?!
        end     += self.n_shared_outs
        index_o += self.n_nit_sot
        while index < end:
            if (hasattr(inputs[index],'dtype') and
                inputs[index].dtype != self.outputs[index_o].dtype):
                raise ValueError(err_msg2%( inputs[index].name
                                           , idx
                                           , inputs[index].dtype
                                           , self.outputs[index_o].dtype) )
            index   += 1
            index_o += 1
        for x in inputs[index:index+ self.n_nit_sot]:
            # For every nit_sot input we get as input a int/uint that
            # depicts the size in memory for that sequence. This feature is
            # used by truncated BPTT and by scan space optimization
            if (str(x.dtype)[:3] not in ('uin','int') or
                x.ndim != 0):
                raise ValueError('For output %d you need to provide a '
                                 'scalar int !',x)

        apply_node = Apply(self
                           , inputs
                           , [t() for t in self.info['output_types']])
        return apply_node

    def __eq__(self, other):
        if not type(self) == type(other):
            return False
        elif not len(self.inputs) == len(other.inputs):
            return False
        elif not len(self.outputs) == len(other.outputs):
            return False
        else:
            for x,y in zip(self.inputs, other.inputs):
                if not scan_utils.equal_computations(x,y):
                    return False
            for x,y in zip(self.outputs, other.outputs):
                if not scan_utils.equal_computations(x,y):
                    return False
            return self.info == other.info

    def __str__(self):
        if self.name:
            return self.name
        else:
            return 'scan'


    def __hash__(self):
        return ( hash(type(self)) ^
                scan_utils.hash_listsDictsTuples(self.inputs) ^
                scan_utils.hash_listsDictsTuples(self.outputs) ^
                scan_utils.hash_listsDictsTuples(self.info) )


    def perform( self, node, args, outs):
        """
        The args are packed like this:

            n_steps

            X sequence inputs x_1, x_2, ... x_<self.n_seqs>

            Y initial states (u_1, u_2, ... u_<self.n_outs>) for our
            outputs. Each must have appropriate length (T_1, T_2, ..., T_Y).

            W other inputs w_1, w_2, ... w_W

        There are at least 1 + self.n_seqs + self.n_outs inputs, and the
        ones above this number are passed to the scanned function as
        non-sequential inputs.

        The outputs are more straightforward:

            Y sequence outputs y_1, y_2, ... y_<self.n_outs>

        """
        # 1. Unzip the number of steps and sequences. If number of steps is
        # negative flip sequences around, and make n_steps positive
        n_steps  = args[0]

        if n_steps < 0:
            n_steps = abs(n_steps)
            seqs = [ seq[::-1] for seq in args[1:self.seqs_arg_offset]]
            seqs = zip( seqs, self.vector_seqs )
        else:
            seqs = args[1:self.seqs_arg_offset]
            seqs = zip( seqs, self.vector_seqs )

        # 2. Allocate memory for the outputs. Construct the list:
        #       store_steps  -- map containting the length of each output
        #       pos          -- map containing the current position of each output

        store_steps  = [ arg.shape[0] for arg
                               in args[self.seqs_arg_offset:
                                       self.shared_arg_offset] ]
        store_steps += [ arg for arg in
                            args[self.nit_sot_arg_offset:
                                   self.nit_sot_arg_offset+self.n_nit_sot]
                       ]

        pos = [ (-self.mintaps[idx])%store_steps[idx] for idx
                         in xrange(self.n_outs+self.n_nit_sot)]
        # 2.1 Create storage space for outputs
        for idx in xrange(self.n_outs):
            if self.inplace:
                # ^ Case 1. Outputs should be computed inplace of their
                # initial state
                outs[idx][0] = args[self.seqs_arg_offset + idx ]
            elif ( outs[idx][0] is not None and
                  outs[idx][0].shape[1:] == args[self.seqs_arg_offset + idx].shape[1:]
                  and outs[idx][0].shape[0] >= store_steps[idx] ):
                # Put in the values of the initial state
                outs[idx][0]       = outs[idx][0][:store_steps[idx]]
                if idx > self.n_mit_mot:
                    l = - self.mintaps[idx]
                    outs[idx][0][:l] = args[self.seqs_arg_offset + idx][:l]
                else:
                    outs[idx][0][:] = args[self.seqs_arg_offset + idx]
            else:
                outs[idx][0] = args[self.seqs_arg_offset + idx].copy()


        offset = self.nit_sot_arg_offset + self.n_nit_sot + self.n_other_ignore
        other_args = args[offset:]
        zipped_outs = [(outs[idx], self.vector_outs[idx], tap,
                       store_steps[idx], idx) for idx in xrange(self.n_outs)
                       for tap in self.tap_array[idx] ]
        end = self.n_outs + self.n_nit_sot
        sot_outs = zip( outs[self.n_mit_mot:end]
                       , self.vector_outs[self.n_mit_mot:end]
                       , store_steps[self.n_mit_mot:end]
                       , range(self.n_mit_mot, end ))

        ############## THE MAIN LOOP #########################
        for i in xrange(n_steps):
            # sequences over which scan iterates
            # 3. collect input slices
            if i == 1 and self.n_nit_sot > 0 :
                sot_outs = zip( outs[self.n_mit_mot:end]
                               , self.vector_outs[self.n_mit_mot:end]
                               , store_steps[self.n_mit_mot:end]
                               , range(self.n_mit_mot, end ))


            fn_args = [ seq[i:i+1].reshape(()) if c else seq[i]
                               for seq,c in seqs]

            fn_args += [ out[0][(pos[j]+tap)%sz:
                                (pos[j]+tap)%sz+1].reshape(())
                        if c else out[0][(pos[j]+tap)%sz]
                        for (out, c, tap, sz, j) in zipped_outs ]
            a_offset = self.shared_arg_offset
            o_offset = self.n_outs + self.n_nit_sot
            fn_args += [ args[a_offset+j] if i==0 else outs[o_offset+j][0]
                        for j in xrange(self.n_shared_outs) ]

            fn_args += other_args

            # 4. collecting slices where the output should be stored
            fn_out_storage = [ [None] for x in xrange(self.n_mit_mot_outs)]
            if i == 0 and self.n_nit_sot > 0:
                fn_out_storage += [
                    [None] if store == 1 or c else [out[0][pos[j]]]
                    for out,c,store,j in sot_outs[:-self.n_nit_sot] ]
                fn_out_storage += [[None]]*self.n_nit_sot
            else:
                fn_out_storage += [
                    [ None ] if store == 1 or c else [out[0][pos[j]]]
                    for out,c,store,j in sot_outs ]

            fn_out_storage += [ [None] for x in xrange(self.n_shared_outs) ]


            # 5. compute outputs
            something = self.fn(fn_args, fn_out_storage)
            offset_out = 0
            # 5.1 Copy over the values for mit_mot outputs
            for j in xrange(self.n_mit_mot):
                for k in self.mit_mot_out_slices[j]:
                    outs[j][0][k+pos[j]] = something[offset_out][0]
                    offset_out += 1

            # 5.2 Copy over the values for mit_sot/sit_sot outputs
            begin = self.n_mit_mot
            end   = self.n_outs
            offset_out -= self.n_mit_mot

            for j in xrange(begin, end):
                if store_steps[j] == 1 or self.vector_outs[j]:
                    outs[j][0][pos[j]] =  something[offset_out+j][0]

            # 5.3 Copy over the values for nit_sot outputs
            begin  = end
            end   += self.n_nit_sot
            for j in xrange(begin,end):
                if i == 0:
                    jout = j+offset_out
                    shape = (store_steps[j],) + something[jout][0].shape
                    if len(something[jout][0].shape) == 0:
                        self.vector_outs[j] = True
                    dtype = something[jout][0].dtype
                    if (outs[j][0] is None or
                        outs[j][0].shape[0] < store_steps[j] or
                        outs[j][0].shape[1:] != shape[1:] or
                        outs[j][0].dtype != dtype ):
                        if self.info['gpu']:
                            outs[j][0] = cuda.cuda_ndarray.cuda_ndarray.CudaNdarray.zeros(shape)
                        else:
                            outs[j][0] = numpy.zeros(shape, dtype)
                    elif outs[j][0].shape[0] != store_steps[j]:
                        outs[j][0] = outs[j][0][:store_steps[j]]
                    outs[j][0][pos[j]] = something[jout][0]
                elif store_steps[j] == 1 or self.vector_outs[j]:
                    outs[j][0][pos[j]] = something[j+offset_out][0]


            # 5.4 Copy over the values for outputs corresponding to shared
            # variables
            begin  = end
            end   += self.n_shared_outs
            for j in xrange(begin,end):
                jout = j +offset_out
                outs[j][0] = something[jout][0]

            pos = [ (idx+1)%store for idx,store in
                               itertools.izip(pos, store_steps)
                               ]


        # 6. Check if you need to re-order output buffers
        begin = self.n_mit_mot
        end   = self.n_outs + self.n_nit_sot
        for idx in xrange(begin, end):
            min_tap = self.mintaps[idx]
            if ( store_steps[idx] < n_steps-self.mintaps[idx] and
                pos[idx] < store_steps[idx] ):
                part_1 = range(pos[idx], store_steps[idx])
                part_2 = range(pos[idx] )
                reordered = part_1 + part_2
                if len(reordered) > 1:
                    if isinstance( outs[idx][0], cuda.CudaNdarray):
                        shape = outs[idx][0].shape
                        tmp = cuda.cuda_ndarray.cuda_ndarray.CudaNdarray.zeros(shape)
                        pdx = pos[idx]
                        tmp[:store_steps[idx]-pdx] = outs[idx][0][pdx:]
                        tmp[store_steps[idx]-pdx:] = outs[idx][0][:pdx]
                        outs[idx][0] = tmp
                    else:
                        outs[idx][0] = outs[idx][0][reordered]
        for idx,val in enumerate(self.mask):
            if val == 1:
                if hasattr(outs[end+idx][0], 'copy'):
                    outs[end + idx][0] = outs[end+idx][0].copy()
                else:
                    outs[end + idx][0] = copy.deepcopy(outs[end+idx][0])


    ### Infer Shape
    def infer_shape(self, node, input_shapes):

        seqs_shape = [ x[1:] for x in input_shapes[1:1+self.n_seqs] ]
        n_outs = self.n_mit_mot + self.n_mit_sot + self.n_sit_sot
        outs_shape = []
        for idx in xrange(n_outs):
            for k in self.tap_array[idx]:
                outs_shape += [ input_shapes[idx+self.n_seqs+1][1:] ]
        offset = 1 + self.n_seqs + n_outs
        for idx in xrange(self.n_shared_outs):
            outs_shape += [ input_shapes[idx+offset] ]

        offset += self.n_nit_sot + self.n_other_ignore + self.n_shared_outs
        inner_ins_shapes = seqs_shape + outs_shape + input_shapes[offset:]
        outs_shape = scan_utils.infer_shape(
            self.outputs
            , self.inputs
            , inner_ins_shapes)
        offset = 1 + self.n_seqs
        scan_outs = [x for x in input_shapes[offset:offset+n_outs]]
        offset += n_outs
        for x in xrange(self.n_nit_sot):
            if outs_shape[n_outs+x] is not None:
                scan_outs.append(
                    (node.inputs[offset+self.n_shared_outs+x],) +
                    tuple(outs_shape[n_outs+x]) )
            else:
                r = node.outputs[n_outs+x]
                shp = (node.inputs[offset+self.n_shared_outs+x],)
                shp += tuple([Shape_i(i)(r) for i in xrange(1,r.ndim)])
                scan_outs.append( shp )
        scan_outs += [ x for x in
                     input_shapes[offset:offset+self.n_shared_outs] ]
        return scan_outs



    ### GRAD FUNCTION
    def grad(self, args, g_outs):
        # 1. forward pass - get the outputs after applying scan
        scan_outputs = self(*args)
        # 2. make sure they are given as a list
        if not( type(scan_outputs) in (list,tuple)):
            scan_outputs = [scan_outputs]
        # 3. un-group / unzip the inputs
        seqs   = self.inputs[:self.n_seqs]

        offset        = self.n_seqs
        n_ins_mit_mot = numpy.sum([0] + [ len(self.tap_array[x]) for x
                                   in xrange(self.n_mit_mot) ])
        outs_mit_mot  = self.inputs[offset:offset+n_ins_mit_mot]

        offset       += n_ins_mit_mot
        n_ins_mit_sot = numpy.sum([0] + [ len(self.tap_array[x]) for x
                                   in xrange( self.n_mit_mot
                                             , self.n_mit_mot+self.n_mit_sot)])
        outs_mit_sot          = self.inputs[offset:offset+n_ins_mit_sot]
        offset               += n_ins_mit_sot
        outs_sit_sot          = self.inputs[offset:offset+self.n_sit_sot]
        offset               += self.n_sit_sot
        old_scan_shared_ins   = self.inputs[offset:offset+self.n_shared_outs]
        out_offset            = ( self.n_mit_mot_outs
                                 + self.n_mit_sot
                                 + self.n_nit_sot
                                 + self.n_sit_sot )
        old_scan_shared_outs  = self.outputs[out_offset:]
        arg_offset = ( 1
                      + self.n_seqs
                      + self.n_mit_mot
                      + self.n_mit_sot
                      + self.n_sit_sot)
        old_scan_init = args[arg_offset: arg_offset+self.n_shared_outs]
        offset       += self.n_shared_outs
        other_args    = self.inputs[offset:]


        # 4. Collect (possibly) differentiable inputs
        diff_inputs = ( seqs          +
                        outs_mit_mot  +
                        outs_mit_sot  +
                        outs_sit_sot  +
                        other_args    )
                       #args[-len(other_args):]    )

        # 5. construct the function that computes the gradient (we sum over
        # the gradients with respect to all outputs)
        def compute_gradient(y, g_y):
            gmp = gradient.grad_sources_inputs(
                        [(y,g_y)], diff_inputs, False )
            return [gmp.get(p, None) for p in diff_inputs ]

        # 6. clean the outputs (i.e. remove update rules)
        end = ( self.n_mit_mot_outs
               + self.n_mit_sot
               + self.n_sit_sot
               + self.n_nit_sot )
        clean_outputs    = self.outputs[:end]
        g_outs_no_shared = g_outs[:end]

        # 7.1. empty lists to hold gradients
        # List of slices from outputs (used to compute the gradients)
        inner_g_outs         = []
        g_out_slices         = []
        # List of outputs of the gradient function
        inner_gfn_outs       = []
        # slices of the input
        prev_inner_gfn_outs  = []
        zeros_like_diff_ins  = []
        pos = ( self.n_seqs + n_ins_mit_mot + n_ins_mit_sot +
               self.n_sit_sot)
        offset = len(args) - len(other_args) - pos
        # 7.2. generate variables to represent previous steps of g_outs
        for idx,diff_in in enumerate(diff_inputs):
            prev_gfn_out = safe_new(diff_in)
            if hasattr(diff_in,'name') and diff_in.name:
                prev_gfn_out.name = 'g_prev_'+diff_in.name
            else:
                prev_gfn_out.name = 'g_prev_'+str(idx)
            prev_inner_gfn_outs.append( prev_gfn_out)
            if idx < pos:
                zeros_like_diff_ins.append(tensor.zeros_like(diff_in))
            else:
                zeros_like_diff_ins.append(tensor.zeros_like(args[idx+offset]))


        # 7.3. compute gradients of the inputs given one output
        for dx, out in enumerate(clean_outputs):
            inner_g_out = safe_new(out)
            if g_outs_no_shared[dx]:
                g_out_slices.append(g_outs_no_shared[dx][0])
            else:
                g_out_slices.append(None)
            if out.name:
                inner_g_out.name = 'g_'+out.name
            else:
                inner_g_out.name = 'g_'+str(dx)
            inner_g_outs.append(inner_g_out)
            _g_out = inner_g_out
            grad_outs = compute_gradient(out, _g_out)
            if not inner_gfn_outs:
                for idx, gfn_out in enumerate(grad_outs):
                    if idx >= self.n_seqs:
                        inner_gfn_outs.append( prev_inner_gfn_outs[idx] )
                    else:
                        inner_gfn_outs.append( None )
            # 7.4 Sum the gradients
            # safety check, some of this inputs might still not be
            # differentiable, for those we don't add them to the mix
            # (assume their gradient is 0)
            for i,(x,y) in enumerate(zip(grad_outs, inner_gfn_outs)):
                if x and y:
                    inner_gfn_outs[i] = x+y
                elif y:
                    inner_gfn_outs[i] = y
                else:
                    inner_gfn_outs[i] = x

        ## 8. Mask the outputs that are not differentiable
        # backwards pass
        for i in xrange(len(inner_gfn_outs)):
            if inner_gfn_outs[i] == None:
                inner_gfn_outs[i] = tensor.zeros_like(diff_inputs[i])

        ## 9. Mask the g_outs that are Nones :
        for i, out in enumerate(scan_outputs):
            if g_outs[i] is None:
                try:
                    # this try is for catching non ndarray inputs (random
                    # states) it is more of a safety check ( all random
                    # states should be after n_outs_not_shared ...
                    g_outs[i] = tensor.zeros_like(scan_outputs[i])
                except:
                    g_outs[i] = theano.tensor.constant(
                        numpy.array(0, theano.config.floatX))


        ## 10. Get your sequence in order for the scan:
        n_seqs  = ( self.n_seqs   +
                   n_ins_mit_mot  +
                   n_ins_mit_sot  +
                   self.n_sit_sot +
                   self.n_nit_sot )
        offset = ( self.n_mit_mot_outs +
                  self.n_mit_sot       +
                  self.n_sit_sot       )
        inner_seqs = ( seqs        +
                      outs_mit_mot +
                      outs_mit_sot +
                      outs_sit_sot +
                      inner_g_outs[offset:offset+self.n_nit_sot])

        scan_seqs = [ x[::-1] for x in args[1:self.n_seqs + 1]]
        offset = 0
        for idx in xrange(self.n_mit_mot + self.n_mit_sot):
            mintap = numpy.min(self.tap_array[idx])
            maxtap = numpy.max(self.tap_array[idx])
            seq    = scan_outputs[offset+idx][::-1]
            for k in self.tap_array[idx]:
                # We cut the sequence such that seq[i] to correspond to
                # seq[i-k]
                if maxtap < 0:
                    dim_offset = abs(maxtap)
                else:
                    dim_offset = 0
                if maxtap == mintap and maxtap != 0:
                    nw_seq =seq[:abs(maxtap)]
                elif maxtap -k != 0 :
                    nw_seq = seq[dim_offset +k -mintap: -(maxtap -k)]
                else:
                    nw_seq = seq[dim_offset +k -mintap: ]
                if seq.name:
                    nw_seq.name = seq.name + '[%d:]'%k
                scan_seqs.append(nw_seq)

        offset += self.n_mit_sot
        for idx in xrange(self.n_sit_sot):
            seq = scan_outputs[offset+idx][:-1]
            scan_seqs.append(seq[::-1])

        offset = ( self.n_mit_mot_outs +
                  self.n_mit_sot       +
                  self.n_sit_sot       )
        scan_seqs += [ x[::-1] for x in
                      g_outs[offset:offset+self.n_nit_sot]]

        scan_mit_mot       = []
        inner_mit_mot      = []
        scan_mit_mot_outs  = []
        mit_mot_taps       = []
        mit_mot_out_slices = []
        out_pos            = 0
        ins_pos            = n_seqs
        n_mit_mot_outs     = 0
        n_mit_mot_ins      = 0
        ins_pos       = self.n_seqs
        for idx in xrange(self.n_mit_mot):
            scan_mit_mot.append( g_outs[idx][::-1] )
            mit_mot_taps.append([])
            mit_mot_out_slices.append([])
            for jdx in xrange(len(self.mit_mot_out_slices[idx])):
                inner_mit_mot.append( inner_g_outs[out_pos] )
                mit_mot_taps[idx].append(
                    -self.mit_mot_out_slices[idx][jdx])
                n_mit_mot_ins += 1
                out_pos       += 1

            for jdx in xrange(len(self.tap_array[idx])):
                inner_mit_mot.append( prev_inner_gfn_outs[ins_pos] )
                scan_mit_mot_outs.append(
                    inner_gfn_outs[ ins_pos] )
                n_mit_mot_ins  += 1
                ins_pos        += 1
                n_mit_mot_outs += 1
                mit_mot_taps[idx].append( -self.tap_array[idx][jdx])
                mit_mot_out_slices[idx].append(
                    -self.tap_array[idx][jdx] )

        offset = self.n_mit_mot
        for idx in xrange(self.n_mit_sot):
            mit_mot_taps.append([])
            mit_mot_out_slices.append([])
            scan_mit_mot.append( g_outs[idx + offset][::-1] )
            idx_tap = idx + self.n_mit_mot
            for jdx in xrange(len(self.tap_array[idx_tap])):
                inner_mit_mot.append( prev_inner_gfn_outs[ins_pos] )
                mit_mot_taps[idx+offset].append(
                    -self.tap_array[idx_tap][jdx] )
                mit_mot_out_slices[idx].append(
                    -self.tap_array[idx_tap][jdx] )
                scan_mit_mot_outs.append(inner_gfn_outs[ ins_pos] )
                n_mit_mot_ins  += 1
                ins_pos        += 1
                n_mit_mot_outs += 1
            inner_mit_mot.append( inner_g_outs[out_pos] )
            out_pos += 1
            n_mit_mot_ins += 1
            mit_mot_taps[idx+offset].append( 0 )

        offset += self.n_mit_sot
        for idx in xrange(self.n_sit_sot):
            mit_mot_taps.append([0,1])
            mit_mot_out_slices.append([1])
            scan_mit_mot.append( g_outs[idx + offset][::-1] )
            scan_mit_mot_outs.append(inner_gfn_outs[ ins_pos ])
            inner_mit_mot += [ inner_g_outs[out_pos]
                              , prev_inner_gfn_outs[ins_pos] ]
            n_mit_mot_outs += 1
            out_pos        += 1
            ins_pos        += 1
            n_mit_mot_ins  += 2


        n_nit_sot = self.n_seqs
        scan_nit_sot_outs = inner_gfn_outs[:self.n_seqs]

        offset = ( self.n_seqs
                  + n_ins_mit_sot
                  + n_ins_mit_mot
                  + self.n_sit_sot )
        n_shared_outs    = len(prev_inner_gfn_outs[offset:])
        scan_shared_ins  = prev_inner_gfn_outs[offset:]
        scan_shared_init = zeros_like_diff_ins[offset:]
        scan_shared_outs = inner_gfn_outs[offset:]
        tap_array        = mit_mot_taps
        info = {}
        info['n_seqs']                   = n_seqs
        info['n_mit_sot']                = 0
        info['tap_array']                = tap_array
        info['gpu']                      = False
        n_mit_mot                        = ( self.n_mit_mot
                                            + self.n_mit_sot
                                            + self.n_sit_sot )
        info['n_mit_mot']                = n_mit_mot
        info['n_mit_mot_outs']           = n_mit_mot_outs
        info['mit_mot_out_slices']       = mit_mot_out_slices
        info['truncate_gradient']        = self.truncate_gradient
        info['n_sit_sot']                = 0
        info['n_shared_outs']            = n_shared_outs + self.n_shared_outs
        info['n_nit_sot']                = n_nit_sot
        if self.name:
            info['name']  = 'grad_of_' + self.name
        else:
            info['name'] = None
        info['mode']                     = self.mode
        info['inplace']                  = False
        info['n_other_ignore']           = 0
        n_mit_sot           = 0
        n_sit_sot           = 0
        n_other_ignore_seqs = 0
        if self.truncate_gradient != -1 :
            do_steps = tensor.minimum(args[0], self.truncate_gradient)
        else:
            do_steps = args[0]

        offset = ( 1
                  + self.n_seqs
                  + self.n_mit_mot
                  + self.n_mit_sot
                  + self.n_sit_sot
                  + self.n_nit_sot
                  + self.n_shared_outs
                  + self.n_other_ignore )

        scan_inputs = ( [do_steps]                            +
                       scan_seqs                              +
                       scan_mit_mot                           +
                       scan_shared_init                       +
                       old_scan_init                          +
                       [ args[0] for x in xrange(n_nit_sot) ] +
                       args[offset:]                          )

        offset = ( self.n_seqs
                  + n_ins_mit_mot
                  + n_ins_mit_sot
                  + self.n_sit_sot
                  + self.n_shared_outs )

        inner_other_args = self.inputs[offset:]
        inner_gfn_ins  = ( inner_seqs         +
                          inner_mit_mot       +
                          scan_shared_ins     +
                          old_scan_shared_ins +
                          inner_other_args )
        inner_gfn_outs = ( scan_mit_mot_outs +
                           scan_nit_sot_outs +
                           scan_shared_outs  +
                           old_scan_shared_outs )

        local_op = Scan( inner_gfn_ins, inner_gfn_outs, info )
        outputs = local_op(*scan_inputs)
        if type(outputs) not in (list, tuple):
            outputs = [ outputs ]
        # Re-order the gradients correctly
        gradients = [None]

        offset = ( self.n_mit_mot
                  + self.n_mit_sot
                  + self.n_sit_sot )
        gradients += [ x[::-1] for x in outputs[offset:offset+self.n_seqs]]

        end = self.n_mit_mot + self.n_mit_sot + self.n_sit_sot
        gradients += [ x[::-1] for x in outputs[:end]]
        gradients += [ None for x in xrange(self.n_shared_outs)]
        gradients += [ None for x in xrange(self.n_nit_sot) ]
        gradients += [ None for x in xrange(self.n_other_ignore) ]
        begin = end + self.n_seqs

        end   = begin + n_shared_outs
        gradients += outputs[begin:end]
        return gradients


@theano.compile.profilemode.register_profiler_printer
def profile_printer(fct_name, compile_time, fct_call_time, fct_call,
                    apply_time, op_cimpl, message, outputs_size,
                    other_time):
    # Scan overhead profile
    if any([isinstance(node.op, Scan) for (_,node) in apply_time.keys()]):
        print
        print 'Scan overhead:'
        print '<Scan op time(s)> <sub scan fct time(s)> <sub scan op time(s)> <sub scan fct time(% scan op time)> <sub scan op time(% scan op time)> <node>'
        total_super_scan_time = 0
        total_scan_fct_time = 0
        total_scan_op_time = 0
        for (_,node),v in apply_time.items():
            if isinstance(node.op, Scan):
                scan_fct_time = sum(node.op.mode_instance.fct_call_time.values())
                scan_op_time = sum(node.op.mode_instance.local_time)
                total_super_scan_time += v
                total_scan_fct_time += scan_fct_time
                total_scan_op_time += scan_op_time
                print '    %5.1fs  %5.1fs  %5.1fs  %5.1f%%  %5.1f%%'%(
                    v, scan_fct_time, scan_op_time, scan_fct_time/v*100,
                    scan_op_time/v*100), node
        print '    total %5.1fs  %5.1fs  %5.1fs  %5.1f%%  %5.1f%%'%(
            total_super_scan_time, total_scan_fct_time, total_scan_op_time, total_scan_fct_time/total_super_scan_time*100, total_scan_op_time/total_super_scan_time*100)
