"""
 This code implements the operations that scan has to carry on when called
 as a stand alone function.

 IF anything this is the entire code that needs to be transported to C.

 Short description of how this code works:
     Scan divides its inputs ( Op's inputs) into different classes of inputs
     as follows:
         i) sequences : inputs over which scan loops to get data. Nothing is
         written into them ( they are readonly, loop over)

         ii) mit_mot : multiple input taps multiple output taps arguments.
         These are inputs over which scan loops and gets data but into which
         scan also writes data. The shorthand mit_mot describes how scan
         deal with them at each step : at each step take several slices as
         input and produce sevaral slices as outputs

         iii) mit_sot : multiple input taps single output tap arguments.
         As before scan reads from these but also writes. At each step scan
         uses several slices as input but produces only one as output

         iv) sit_sot : single input tap single output tap arguments.
         At each step use only the previous slice as input, produce only one
         slice as output

         v) nit_sot: no input tap single output tap arguments.
         At each step don't use any previous values, only produce new onese

         vi) shared_outs: arguments corresponding to shared variables with
         updates.
         At each step use its value as input, and afterwards replace it with
         a new value.
         vii) other_args: arguments that are passed to every call of the
         inner function as they are ( no slicing is perfomed)

    All these outputs are one after the other in the inputs list (named in
    this code as args) in a given order ( namely the one described above
    with little discrepencies depending if we are talking about the outputs
    of the Scan op or the inputs of the Scan op Node, and if we are talking
    about the inputs of the inner function of scan or of the scan op).

    Because of this, all we need to be able to separate and tell arguments
    apart is how many of which we have as well as how many taps and which
    ones (where applicable). All this information is desribed (more or less)
    by describing the arguments of this function)
"""


__authors__ = "Razvan Pascanu"
__copyright__ = "(c) 2011, Universite de Montreal"
__contact__ = "Razvan Pascanu <r.pascanu@gmail>"


import cython

import numpy
cimport numpy
from theano import gof
import time
import copy


def get_version():
    return 0.293

@cython.boundscheck(False)
def perform(
            unsigned int n_shared_outs,
            unsigned int n_mit_mot_outs,
            unsigned int n_seqs,
            unsigned int n_mit_mot,
            unsigned int n_mit_sot,
            unsigned int n_sit_sot,
            unsigned int n_nit_sot,
            int n_steps,
            bint as_while,
            numpy.ndarray[numpy.int32_t,ndim=1] mintaps,
            numpy.ndarray[numpy.int32_t,ndim=2] tap_array,
            numpy.ndarray[numpy.int32_t,ndim=1] tap_array_len,
            numpy.ndarray[numpy.int32_t,ndim=1] vector_seqs,
            numpy.ndarray[numpy.int32_t,ndim=1] vector_outs,
            numpy.ndarray[numpy.int32_t,ndim=2] mit_mot_out_slices,
            numpy.ndarray[numpy.int32_t,ndim=1] mit_mot_out_nslices,
            numpy.ndarray[numpy.int32_t,ndim=1] mitmots_preallocated,
            numpy.ndarray[numpy.int32_t,ndim=1] inps_is_tensor,
            numpy.ndarray[numpy.int32_t,ndim=1] outs_is_tensor,
            fn,
            fnct,
            numpy.ndarray[numpy.int32_t,ndim=1] destroy_map,
            args,
            outs,
            self,
            node):
    """
    Parameters
    ----------
    n_shared_outs: unsigned int
        Number of arugments that correspond to shared variables with
        updates
    n_mit_mot_outs: unsigned int
        Sum over the number of output taps for each mit_mot sequence
    n_seqs: unsigned int
        Number of sequences provided as input
    n_mit_mot : unsigned int
        Number of mit_mot arguemnts
    n_mit_sot: unsigned int
        Number of mit_sot arguments
    n_sit_sot: unsigned int
        Number of sit sot arguemnts
    n_nit_sot: unsigned int
        Number of nit_sot arguments
    n_steps: unsigned int
        Number of steps to loop over
    mintaps: int32 ndarray (can also be a simple python list if that is better !)
        For any of the mit_mot, mit_sot, sit_sot says which is the furtherst
        away input tap from current position. For example, if the taps where [-2,
        -5, -9], the mintap would be -9. For sit_sot this is always -1 since
        is the only allowed tap.
    tap_array: int32 ndarray( can be replaced by a list of list in python if better)
        For each of the mit_mot, mit_sot, sit_sot (the first dimension) says
        which are the corresponding input taps. While this is a matrix, not all
        values in a row are needed and tap_array_len is there to say up to
        which entry we are dealing with valid taps ( afterwards there are
        just 0s to ensure the fix format)
    tap_array_len: int32 ndarray( can be replaced by a list if better)
        For each of the mit_mot, mit_sot, sit_sot says how many input taps
        each has. For sit_sot this will always be 1.
    vector_seqs: int32 ndarray (can be replaced by a list of bools if better)
        For each sequence the corresponding entry is either a 1, is the
        sequence is a vector or 0 if it has more than 1 dimension
    vector_outs: int32 ndarray( can be replaced by list of bools if better)
        For each output ( mit_mot, mit_sot, sit_sot, nit_sot in this order)
        the entry is 1 if the corresponding argument is a 1 dimensional
        tensor, 0 otherwise.
    mit_mot_out_slices : int32 ndarray( can be replaced by list of lists)
        Same as tap_array, but for the output taps of mit_mot sequences
    mit_mot_out_nslices: int32 ndarray (Can be replaced by a list)
        Same as tap_array_len, but is the number of output taps of the
        mit_mot sequences (i.e. it corresponds to mit_mot_out_slices)
    inps_is_tensor : int32 ndarray (Can be replaced by a list)
        Array of boolean indicating, for every input, whether it is a tensor
        or not
    outs_is_tensor : int32 ndarray (Can be replaced by a list)
        Array of boolean indicating, for every output, whether it is a tensor
        or not
    fn: callable
        This is the linker, i.e. the function that will loop over the
        computational graph and call the perform of each operation. For this
        linker there is a c version in gof/lazy_linker.c that will be the
        starting point of implementing this funciton in C ( we need to take
        all the code around the call of this function and put in C inside
        that code)
    fnct: python object
        Only used to attach some timings for the profile mode ( can be
        skiped if we don't care about Theano's profile mode)
    destroy_map
        Array of boolean saying if an output is computed inplace
    args: list of ndarrays (and random states)
        The inputs of scan in a given order ( n_steps, sequences, mit_mot,
        mit_sot, sit_sot, nit_sot, shared_outs, other_args)
    outs: list of 1 element list ( or storage objects?)
        This is where we need to copy our outputs ( we don't return the
        results, though we can change the code such that we return, and
        figure things out on the outside - python)
    self: python object
        The scan op itself. I only use it to attach to it some timing
        informations .. but I don;t need to.

    """
    # 1. Unzip the number of steps and sequences. If number of steps is
    # negative flip sequences around, and make n_steps positive
    t0_call = time.time()
    t_fn = 0
    cdef unsigned int n_outs = n_mit_mot + n_mit_sot + n_sit_sot
    cdef unsigned int seqs_arg_offset = n_seqs + 1
    cdef unsigned int shared_arg_offset = ( 1 + n_seqs + n_mit_mot +
                                           n_mit_sot + n_sit_sot)
    cdef unsigned int nit_sot_arg_offset = ( shared_arg_offset +
                                            n_shared_outs)
    cdef unsigned int offset_out
    cdef unsigned int lenpos = n_outs + n_nit_sot
    cdef int pos[500] # put a maximum of 500 outputs
    cdef unsigned int len_store_steps = n_mit_mot + n_mit_sot + n_sit_sot + n_nit_sot
    cdef int store_steps[500]
    cdef unsigned int l
    cdef unsigned int offset
    cdef int tap
    cdef int _idx
    cdef unsigned int a_offset
    cdef unsigned int o_offset
    cdef unsigned int idx
    cdef unsigned int i
    cdef unsigned int j
    cdef int k
    cdef unsigned int kdx
    cdef unsigned int tdx
    cdef unsigned int pdx
    cdef unsigned int jout
    cdef unsigned int begin
    cdef unsigned int end
    cdef int cond
    cdef unsigned int len_output_storage = (n_mit_mot_outs + n_mit_sot +
                                            n_sit_sot + n_nit_sot +
                                            n_shared_outs)


    if n_steps < 0:
        # History, in the past, this was used for backward
        # scan. Now we reverse the inputs outside of scan.
        raise IndexError(
            "Scan was asked to run for negative number of step %d" %
            n_steps)
    elif n_steps == 0:
        raise NotImplementedError(
            "We didn't implemented yet the case where scan do 0 iteration")
    else:
        for idx in range(n_seqs):
            if args[<unsigned int>(1+idx)].shape[0] < n_steps:
                raise ValueError(('Sequence is shorter then the required '
                                 'number of steps : (n_steps, seq, '
                                  'seq.shape):'), n_steps,
                                  args[1+idx],
                                  args[1+idx].shape)
    # 2. Allocate memory for the outputs. Construct the list:
    #       store_steps  -- map containting the length of each output
    #       pos          -- map containing the current position of each output

    for idx in range(n_mit_mot + n_mit_sot + n_sit_sot):
        store_steps[<unsigned int>idx] = args[<unsigned int>(idx+n_seqs+1)].shape[0]

    for idx in range(n_nit_sot):
        store_steps[<unsigned int>(idx + n_mit_mot + n_mit_sot + n_sit_sot)]=\
                args[<unsigned int>(idx + n_mit_mot + n_mit_sot + n_sit_sot
                                    + n_shared_outs + n_seqs+1)]

    for idx in range(n_outs + n_nit_sot):
        pos[idx] = (-mintaps[idx])%store_steps[idx]


    # 2.1 Create storage space for outputs
    for idx in range(n_outs):
        if destroy_map[idx] != 0:
            # ^ Case 1. Outputs should be computed inplace of their
            # initial state
            outs[idx][0] = args[ <unsigned int>(1+ n_seqs + idx)]
        elif ( outs[idx][0] is not None and
              outs[idx][0].shape[1:] == args[<unsigned int>(1+ n_seqs + idx)].shape[1:]
              and outs[idx][0].shape[0] >= store_steps[idx] ):
            # Put in the values of the initial state
            outs[idx][0] = outs[idx][0][:store_steps[idx]]
            if idx > n_mit_mot:
                l = - mintaps[idx]
                outs[idx][0][:l] = args[<unsigned int>(seqs_arg_offset +
                                                       idx)][:l]
            else:
                outs[idx][0][:] = args[<unsigned int>(seqs_arg_offset + idx)]
        else:
            outs[idx][0] = args[<unsigned int>(seqs_arg_offset + idx)].copy()


    offset = nit_sot_arg_offset + n_nit_sot
    other_args = args[offset:]
    input_storage = fnct.input_storage
    nb_mitmot_in = 0
    for idx in range(n_mit_mot):
        nb_mitmot_in += tap_array_len[idx]
    old_mitmot_input_storage = [None] * nb_mitmot_in
    old_mitmot_input_data = [None] * nb_mitmot_in
    output_storage = fnct.output_storage
    old_output_storage = [None] * len_output_storage
    old_output_data = [None] * len_output_storage
    offset = n_seqs
    for idx in range(n_outs):
        offset += tap_array_len[idx]
    offset += n_shared_outs

    for idx in range(len(other_args)):
        input_storage[<unsigned int>(idx+offset)].storage[0] = other_args[idx]


    i = 0
    cond = 1
    ############## THE MAIN LOOP #########################
    #for i in range(n_steps):
    while (i < n_steps) and cond == 1:
        # sequences over which scan iterates
        # 3. collect input slices
        for idx in range(n_seqs):
            if vector_seqs[idx] == 1:
                input_storage[idx].storage[0] = args[\
                            <unsigned int>(1+idx)][i:<unsigned int>(i+1)].reshape(())
            else:
                input_storage[idx].storage[0] = \
                        args[<unsigned int>(idx+1)][i]

        offset = n_seqs
        for idx in range(n_outs):
            if vector_outs[idx] == 1:
                for tdx in range(tap_array_len[idx]):
                    tap = tap_array[idx,tdx]
                    _idx = (pos[idx]+tap)%store_steps[idx]
                    input_storage[offset].storage[0] =\
                            outs[idx][0][_idx:<unsigned int>(_idx+1)].reshape(())
                    offset += 1
            else:
                for tdx in range(tap_array_len[idx]):
                    tap = tap_array[idx,tdx]
                    _idx = (pos[idx]+tap)%store_steps[idx]
                    input_storage[offset].storage[0] = outs[idx][0][_idx]
                    offset += 1


        a_offset = shared_arg_offset
        o_offset = n_outs + n_nit_sot
        if i == 0:
            for j in range(n_shared_outs):
                input_storage[offset].storage[0] = args[<unsigned int>(a_offset+j)]
                offset += 1
        else:
            for j in range(n_shared_outs):
                input_storage[offset].storage[0] = outs[<unsigned int>(o_offset+j)][0]
                offset += 1

        # 4. collecting slices where the output should be stored

        # 4.1. Collect slices for mitmots
        offset = 0
        for idx in range(n_mit_mot_outs):
            if not mitmots_preallocated[<unsigned int>idx]:
                output_storage[<unsigned int>offset].storage[0] = None
                offset += 1

        # 4.2. Collect slices for mitsots, sitsots and nitsots
        if i != 0:
            for idx in range(n_outs + n_nit_sot - n_mit_mot):
                if ( store_steps[<unsigned int>(idx+n_mit_mot)] == 1 or
                    vector_outs[<unsigned int>(idx+n_mit_mot)] == 1):
                    output_storage[<unsigned int>(idx+offset)].storage[0] = None
                else:
                    output_storage[<unsigned int>(idx+offset)].storage[0] =\
                        outs[<unsigned int>(idx+n_mit_mot)][0][pos[\
                                            <unsigned int>(idx+n_mit_mot)]]
        else:
            for idx in range(n_outs + n_nit_sot - n_mit_mot):
                output_storage[<unsigned int>(idx+offset)].storage[0] = None

        # 4.3. Collect slices for shared outputs
        offset += n_outs+n_nit_sot - n_mit_mot
        for idx in range(n_shared_outs):
            output_storage[<unsigned int>(idx+offset)].storage[0] = None

        # 4.4. If there is a condition add it to the mix
        if as_while:
            pdx = offset + n_shared_outs
            output_storage[<unsigned int>pdx].storage[0] = None

        # 4.5. Keep a reference to the variables (ndarrays, CudaNdarrays,
        # etc) currently in the output_storage to be able to compare them
        # with the actual outputs of the inner function after its
        # execution. Also keep pointers to their data to be able to detect
        # cases where outputs reused the allocated object but alter the
        # memory region they refer to.
        for idx in range(len_output_storage):

            var = output_storage[idx].storage[0]
            old_output_storage[idx] = var

            if var is None:
                old_output_data[idx] = None
            elif outs_is_tensor[idx]:
                old_output_data[idx] = var.data
            else:
                old_output_data[idx] = var.gpudata

        # 4.6. Keep a reference to the variables (ndarrays, CudaNdarrays,
        # etc) associated with mitmot inputs currently in the input_storage to
        # be able to compare them with the content of the input_storage after
        # the execution of the function. Also keep pointers to their data to
        # be able to detect cases where outputs reused the allocated object
        # but alter the memory region they refer to.
        for idx in xrange(nb_mitmot_in):
            var = input_storage[idx + n_seqs].storage[0]
            old_mitmot_input_storage[idx] = var

            if var is None:
                old_mitmot_input_data[idx] = None
            elif inps_is_tensor[idx + n_seqs]:
                old_mitmot_input_data[idx] = var.data
            else:
                old_mitmot_input_data[idx] = var.gpudata

        # 5.1 compute outputs
        t0_fn = time.time()

        try:
            fn()
        except Exception:
            if hasattr(fn, 'position_of_error'):
                # this is a new vm-provided function
                # the C VM needs this because the exception manipulation
                # done by raise_with_op is not implemented in C.
                if hasattr(fn, 'thunks'):
                    # For the CVM
                    gof.link.raise_with_op(fn.nodes[fn.position_of_error],
                                           fn.thunks[fn.position_of_error])
                else:
                    # For the c linker
                    # We don't have access from python to all the
                    # temps values So for now, we just don't print
                    # the extra shapes/strides info
                    gof.vm.raise_with_op(fn.nodes[fn.position_of_error])
            else:
                # old-style linkers raise their own exceptions
                raise

        dt_fn = time.time() - t0_fn
        t_fn += dt_fn
        if self.as_while:
            pdx = offset + n_shared_outs
            cond = output_storage[pdx].storage[0] == 0

        # 5.2. By calling fn() directly instead of calling the theano
        # function, it is possible that the updates have not been
        # performed. Perform the updates if needed.
        offset_out = len(output_storage) - 1
        if getattr(fn, 'need_update_inputs', True):
            # Update the inputs that have an update function
            for inp, storage in zip(self.fn.maker.expanded_inputs[::-1],
                                    self.fn.input_storage[::-1]):
                if inp.update is not None:
                    storage.data = output_storage[offset_out].data
                    offset_out -= 1

        offset_out = 0

        # 5.3 Copy over the values for mit_mot outputs
        mitmot_inp_offset = 0
        mitmot_out_idx = 0
        for j in xrange(self.n_mit_mot):
            for k in self.mit_mot_out_slices[j]:
                if mitmots_preallocated[<unsigned int>mitmot_out_idx]:
                    # This output tap has been preallocated.
                    inp_idx = (mitmot_inp_offset +
                               self.tap_array[j].index(k))

                    # Verify whether the input points to the same data as
                    # it did before the execution of the inner function.
                    old_var = old_mitmot_input_storage[inp_idx]
                    new_var = input_storage[n_seqs + inp_idx].storage[0]
                    if old_var is new_var:
                        old_data = old_mitmot_input_data[inp_idx]
                        if inps_is_tensor[n_seqs + inp_idx]:
                            same_data = (new_var.data == old_data)
                        else:
                            same_data = (new_var.gpudata == old_data)
                    else:
                        same_data = False

                    # If the corresponding input storage has been replaced,
                    # recover the value as usual. Otherwise, the input was
                    # modified inplace and nothing needs to be done.
                    if not same_data:
                        outs[j][0][<unsigned int>(k + pos[j])] = \
                            input_storage[<unsigned int>(n_seqs + inp_idx)].storage[0]

                else:
                    # This output tap has not been preallocated, recover
                    # its value as usual
                    outs[j][0][<unsigned int>(k + pos[j])] = \
                            output_storage[<unsigned int>offset_out].storage[0]
                    offset_out += 1

                mitmot_out_idx += 1

            mitmot_inp_offset += len(self.tap_array[j])

        # 5.4 Copy over the values for mit_sot/sit_sot outputs
        begin = n_mit_mot
        end   = n_outs
        offset_out -= n_mit_mot

        for j in range(begin, end):

            # Copy the output value to `outs`, if necessary
            if store_steps[j] == 1 or vector_outs[j] == 1:
                outs[j][0][pos[j]] = output_storage[<unsigned int>(offset_out+j)].storage[0]
            else:
                # Check whether the initialization of the output storage map
                # for this output has been reused.
                old_var = old_output_storage[offset_out + j]
                old_data = old_output_data[offset_out + j]
                new_var = output_storage[offset_out + j].storage[0]
                if old_var is new_var:
                    if old_data is None:
                        output_reused = False
                    elif outs_is_tensor[offset_out + j]:
                        output_reused = (new_var.data == old_data)
                    else:
                        output_reused = (new_var.gpudata == old_data)
                else:
                    output_reused = False

                if not output_reused:
                    outs[j][0][pos[j]] = \
                        output_storage[<unsigned int>(offset_out+j)].storage[0]


        # 5.5 Copy over the values for nit_sot outputs
        begin  = end
        end   += n_nit_sot
        for j in range(begin,end):

            if i == 0:
                jout = j+offset_out
                shape = (store_steps[j],) + output_storage[jout].storage[0].shape
                if len(output_storage[jout].storage[0].shape) == 0:
                    vector_outs[j] = 1
                dtype = output_storage[jout].storage[0].dtype
                if (outs[j][0] is None or
                        outs[j][0].shape[0] < store_steps[j] or
                        outs[j][0].shape[1:] != shape[1:] or
                        outs[j][0].dtype != dtype ):
                    outs[j][0] = node.outputs[j].type.value_zeros(shape)
                elif outs[j][0].shape[0] != store_steps[j]:
                    outs[j][0] = outs[j][0][:store_steps[j]]
                outs[j][0][pos[j]] = output_storage[jout].storage[0]
            elif store_steps[j] == 1 or vector_outs[j] == 1:
                outs[j][0][pos[j]] = output_storage[j+offset_out].storage[0]
            else:
                # Check whether the initialization of the output storage map
                # for this output has been reused.
                old_var = old_output_storage[offset_out + j]
                old_data = old_output_data[offset_out + j]
                new_var = output_storage[offset_out + j].storage[0]
                if old_var is new_var:
                    if old_data is None:
                        output_reused = False
                    elif outs_is_tensor[offset_out + j]:
                        output_reused = (new_var.data == old_data)
                    else:
                        output_reused = (new_var.gpudata == old_data)
                else:
                    output_reused = False

                if not output_reused:
                    outs[j][0][pos[j]] = output_storage[j+offset_out].storage[0]

        # 5.6 Copy over the values for outputs corresponding to shared
        # variables
        begin  = end
        end   += n_shared_outs
        for j in range(begin,end):
            jout = j +offset_out
            outs[j][0] = output_storage[jout].storage[0]

        for idx in range(lenpos):
            pos[idx] = (pos[idx]+1)%store_steps[idx]
        i = i + 1

    # 6. Check if you need to re-order output buffers
    begin = n_mit_mot
    end   = n_outs + n_nit_sot
    for idx in range(begin, end):
        if ( store_steps[idx] < i-mintaps[idx] and
            pos[idx] < store_steps[idx] ):

            pdx = pos[idx]
            if pdx >= store_steps[idx]//2 :
                # It seems inefficient to copy the bigger part of the
                # array over, and back, but it is the only way that
                # there is no overlap in the areas of out[idx][0] that
                # are read and written.
                # This way, there will be no information overwritten
                # before it is read (as it used to happen).
                shape = (pdx,)+ outs[idx][0].shape[1:]

                tmp = node.outputs[idx].type.value_zeros(shape)
                tmp[:] = outs[idx][0][:pdx]
                outs[idx][0][:store_steps[idx]-pdx] = outs[idx][0][pdx:]
                outs[idx][0][store_steps[idx]-pdx:] = tmp
            else:
                shape = (store_steps[idx]-pdx,) + outs[idx][0].shape[1:]
                tmp = node.outputs[idx].type.value_zeros(shape)
                tmp[:] = outs[idx][0][pdx:]
                outs[idx][0][store_steps[idx]-pdx:] = outs[idx][0][:pdx]
                outs[idx][0][:store_steps[idx]-pdx] = tmp
        # This would normally happen only when doing truncated
        # backpropagation through time. In such a scenarion Scan is
        # expected to return 0 for all entries for which the gradient is
        # not actually computed
        elif store_steps[idx] > i - self.mintaps[idx]:
            outs[idx][0][i-self.mintaps[idx]:] = 0

            # This is a fix for a bug introduced by while. If you say
            # you want to loop up to a condition, you expect the output
            # to have that length ( and not the maximal length possible)
            #
            # Without this the behaviour of a scan op is not consistent
            # if optimization gets applied compared to when optimization
            # do not get applied
            if i < n_steps:

	    # Cython can not handle negative indices ( because of a
	    # derictive at the begining of the function that says not
	    # to do boundschecks). The directive is used to make the
	    # code faster, so this workaround is better then removing
	    # the directive.
                sh0 = outs[idx][0].shape[0]
                outs[idx][0] = outs[idx][0][:sh0-(n_steps - i)]

    # We never reuse the input or output storage of the
    # inner function so we clear it.
    for i_s in input_storage:
        i_s.storage[0] = None
    for o_s in output_storage:
        o_s.storage[0] = None

    t_call = time.time() - t0_call

    if hasattr(fnct.maker, 'profile'):
        profile = fnct.maker.profile
        if type(profile) is not bool and profile:
            profile.vm_call_time +=  t_fn
            profile.callcount += 1
            profile.nbsteps += n_steps
            profile.call_time += t_call
            if hasattr(fn, 'update_profile'):
                fn.update_profile(profile)

    ### Old Profile Mode
    #if hasattr(fnct.maker.mode,'fct_call_time'):
    #    fnct.maker.mode.fct_call_time[fnct] += t_fn
    #    fnct.maker.mode.fct_call[fnct] += n_steps

    #fnct.maker.mode.call_time += t_fn
    #fnct.maker.mode.fn_time += t_fn

    # DEBUG PRINT :
    self.t_call = t_call
    self.t_fn   = t_fn
    # print 'Cython > timing', t_call, t_fn, 'in percentage', 100.*t_fn/t_call

