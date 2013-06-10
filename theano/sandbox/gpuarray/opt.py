from theano.compile import optdb
from theano.gof import (local_optimizer, EquilibriumDB, SequenceDB, ProxyDB,
                        Optimizer, toolbox, DestroyHandler,
                        InconsistencyError, EquilibriumOptimizer)

from theano.gof.python25 import all, any
from theano.sandbox.gpuarray.type import GpuArrayType

gpu_optimizer = EquilibriumDB()
gpu_cut_copies = EquilibriumDB()

gpu_seqopt = SequenceDB()

gpu_seqopt.register('gpuarray_local_optimiziations', gpu_optimizer, 1,
                    'fast_run', 'inplace', 'gpuarray')
gpu_seqopt.register('gpuarray_cut_transfers', gpu_cut_copies, 2,
                    'fast_run', 'gpuarray')

optdb.register('gpuarray_opt', gpu_seqopt,
               optdb.__position__.get('add_destroy_handler', 49.5) - 1,
               'gpu')

optdb.register('gpuarray_after_fusion', ProxyDB(gpu_seqopt),
               optdb.__position__.get('elemwise_fusion', 71) + 1,
               'gpu')

def register_opt(*tags, **kwargs):
    def f(local_opt):
        name = (kwargs and kwargs.pop('name')) or local_opt.__name__
        gpu_optimizer.register(name, local_opt, 'fast_run', 'gpu', *tags)
        return local_opt
    return f

register_opt()(theano.tensor.opt.local_track_shape_i)

class InputToGpuOptimizer(Optimizer):
    "Transfer the input to the gpu to start the rolling wave."

    def add_requirements(self, fgraph):
        fgraph.attach_feature(toolbox.ReplaceValidate())
        fgraph.attach_feature(DestroyHandler())

    def apply(self, fgraph):
        for input in fgraph.inputs:
            if isinstance(input.type, GpuArrayType):
                return
            
            if (len(input.clients) == 1 and
                (input.clients[0][0] == 'output' or
                 input.clients[0][0].op == gpu_from_host)):
                return

            try:
                new_input = host_from_gpu(gpu_from_host(input))
                fgraph.replace_validate(input, new_input,
                                        "InputToGpuOptimizer")
            except TypeError, e:
                # This could fail if the inputs are not TensorTypes
                pass

gpu_seqopt.register('InputToGpuArrayOptimizer', InputToGpuOptimizer(),
                    0, 'fast_run', 'fast_compile', 'merge')

@local_optimizer([])
def local_cut_gpu_host_gpu(node):
    if tensor.opt.opt.check_chain(node, gpu_from_host, host_from_gpu):
        return [node.inputs[0].owner.inputs[0]]
    if tensor.opt.opt.check_chain(node, host_from_gpu, gpu_from_host):
        return [node.inputs[0].owner.inputs[0]]
    return False
gpu_cut_copies.register('cut_gpu_host_transfers', local_cut_gpu_host_gpu,
                        'fast_run', 'inplace', 'gpu')
gpu_cut_copies.register('cut_gpu_constant_transfers',
                        tensor.opt.constant_folding,
                        'fast_run', 'gpu')
optdb['canonicalize'].register('local_cut_gpu_host_gpu',
                               local_cut_gpu_host_gpu, 'fast_run', 'gpu')
