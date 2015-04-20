"""
Test flake8 errors.
"""

from nose.plugins.skip import SkipTest
import os
from fnmatch import fnmatch
import theano
try:
    import flake8.engine
    import flake8.main
    flake8_available = True
except ImportError:
    flake8_available = False

__authors__ = ("Saizheng Zhang")
__copyright__ = "(c) 2015, Universite de Montreal"
__contact__ = "Saizheng Zhang <saizhenglisa..at..gmail.com>"

whitelist_flake8 = [
    "__init__.py",
    "version.py",
    "tests/test_gradient.py",
    "tests/test_config.py",
    "tests/diverse_tests.py",
    "tests/test_ifelse.py",
    "tests/test_rop.py",
    "tests/test_2nd_order_grads.py",
    "tests/run_tests_in_batch.py",
    "tests/test_record.py",
    "tests/__init__.py",
    "tests/test_updates.py",
    "tests/main.py",
    "tests/test_pickle_unpickle_theano_fn.py",
    "tests/test_determinism.py",
    "tests/record.py",
    "tests/test_printing.py",
    "tests/test_tutorial.py",
    "tests/disturb_mem.py",
    "tests/unittest_tools.py",
    "compile/ops.py",
    "compile/debugmode.py",
    "compile/function.py",
    "compile/pfunc.py",
    "compile/mode.py",
    "compile/profilemode.py",
    "compile/builders.py",
    "compile/__init__.py",
    "compile/profiling.py",
    "compile/function_module.py",
    "compile/sharedvalue.py",
    "compile/monitormode.py",
    "compile/io.py",
    "compile/module.py",
    "compile/tests/test_builders.py",
    "compile/tests/test_misc.py",
    "compile/tests/test_monitormode.py",
    "compile/tests/test_function_module.py",
    "compile/tests/test_inplace_opt_for_value.py",
    "compile/tests/test_shared.py",
    "compile/tests/test_ops.py",
    "compile/tests/test_pfunc.py",
    "compile/tests/test_module.py",
    "compile/tests/test_debugmode.py",
    "compile/tests/test_profiling.py",
    "typed_list/type.py",
    "typed_list/__init__.py",
    "typed_list/opt.py",
    "typed_list/basic.py",
    "typed_list/tests/test_type.py",
    "typed_list/tests/test_opt.py",
    "typed_list/tests/test_basic.py",
    "tensor/var.py",
    "tensor/sharedvar.py",
    "tensor/inplace.py",
    "tensor/slinalg.py",
    "tensor/shared_randomstreams.py",
    "tensor/subtensor.py",
    "tensor/elemwise.py",
    "tensor/xlogx.py",
    "tensor/blas_headers.py",
    "tensor/utils.py",
    "tensor/type.py",
    "tensor/fourier.py",
    "tensor/sort.py",
    "tensor/__init__.py",
    "tensor/opt_uncanonicalize.py",
    "tensor/opt.py",
    "tensor/blas.py",
    "tensor/extra_ops.py",
    "tensor/nlinalg.py",
    "tensor/blas_c.py",
    "tensor/io.py",
    "tensor/elemwise_cgen.py",
    "tensor/raw_random.py",
    "tensor/randomstreams.py",
    "tensor/blas_scipy.py",
    "tensor/basic.py",
    "tensor/tests/test_subtensor.py",
    "tensor/tests/test_utils.py",
    "tensor/tests/test_nlinalg.py",
    "tensor/tests/test_randomstreams.py",
    "tensor/tests/test_shared_randomstreams.py",
    "tensor/tests/test_misc.py",
    "tensor/tests/test_naacl09.py",
    "tensor/tests/mlp_test.py",
    "tensor/tests/test_opt_uncanonicalize.py",
    "tensor/tests/test_opt.py",
    "tensor/tests/test_basic.py",
    "tensor/tests/test_blas.py",
    "tensor/tests/test_elemwise.py",
    "tensor/tests/test_merge.py",
    "tensor/tests/test_gc.py",
    "tensor/tests/test_complex.py",
    "tensor/tests/test_type_other.py",
    "tensor/tests/test_io.py",
    "tensor/tests/test_sharedvar.py",
    "tensor/tests/test_fourier.py",
    "tensor/tests/test_casting.py",
    "tensor/tests/test_sort.py",
    "tensor/tests/test_raw_random.py",
    "tensor/tests/test_xlogx.py",
    "tensor/tests/test_extra_ops.py",
    "tensor/tests/test_slinalg.py",
    "tensor/tests/test_blas_c.py",
    "tensor/tests/test_blas_scipy.py",
    "tensor/tests/test_mpi.py",
    "tensor/signal/downsample.py",
    "tensor/signal/conv.py",
    "tensor/signal/tests/test_conv.py",
    "tensor/signal/tests/test_downsample.py",
    "tensor/nnet/nnet.py",
    "tensor/nnet/Conv3D.py",
    "tensor/nnet/__init__.py",
    "tensor/nnet/ConvTransp3D.py",
    "tensor/nnet/sigm.py",
    "tensor/nnet/ConvGrad3D.py",
    "tensor/nnet/conv3d2d.py",
    "tensor/nnet/conv.py",
    "tensor/nnet/neighbours.py",
    "tensor/nnet/tests/test_conv.py",
    "tensor/nnet/tests/test_neighbours.py",
    "tensor/nnet/tests/test_nnet.py",
    "tensor/nnet/tests/test_conv3d2d.py",
    "tensor/nnet/tests/test_conv3d.py",
    "tensor/nnet/tests/speed_test_conv.py",
    "tensor/nnet/tests/test_sigm.py",
    "scalar/sharedvar.py",
    "scalar/basic_scipy.py",
    "scalar/basic_sympy.py",
    "scalar/__init__.py",
    "scalar/basic.py",
    "scalar/tests/test_basic.py",
    "sandbox/test_theano_object.py",
    "sandbox/test_scan.py",
    "sandbox/rng_mrg.py",
    "sandbox/downsample.py",
    "sandbox/solve.py",
    "sandbox/theano_object.py",
    "sandbox/scan.py",
    "sandbox/multinomial.py",
    "sandbox/neighbourhoods.py",
    "sandbox/fourier.py",
    "sandbox/test_multinomial.py",
    "sandbox/minimal.py",
    "sandbox/test_rng_mrg.py",
    "sandbox/test_neighbourhoods.py",
    "sandbox/symbolic_module.py",
    "sandbox/conv.py",
    "sandbox/debug.py",
    "sandbox/neighbours.py",
    "sandbox/cuda/dnn.py",
    "sandbox/cuda/var.py",
    "sandbox/cuda/GpuConvGrad3D.py",
    "sandbox/cuda/basic_ops.py",
    "sandbox/cuda/nnet.py",
    "sandbox/cuda/elemwise.py",
    "sandbox/cuda/type.py",
    "sandbox/cuda/__init__.py",
    "sandbox/cuda/opt.py",
    "sandbox/cuda/blas.py",
    "sandbox/cuda/blocksparse.py",
    "sandbox/cuda/GpuConv3D.py",
    "sandbox/cuda/extra_ops.py",
    "sandbox/cuda/rng_curand.py",
    "sandbox/cuda/fftconv.py",
    "sandbox/cuda/kernel_codegen.py",
    "sandbox/cuda/GpuConvTransp3D.py",
    "sandbox/cuda/nvcc_compiler.py",
    "sandbox/cuda/neighbours.py",
    "sandbox/cuda/tests/walltime.py",
    "sandbox/cuda/tests/test_fftconv.py",
    "sandbox/cuda/tests/test_gradient.py",
    "sandbox/cuda/tests/test_neighbours.py",
    "sandbox/cuda/tests/test_conv_cuda_ndarray.py",
    "sandbox/cuda/tests/test_type.py",
    "sandbox/cuda/tests/test_var.py",
    "sandbox/cuda/tests/test_opt.py",
    "sandbox/cuda/tests/test_blas.py",
    "sandbox/cuda/tests/test_driver.py",
    "sandbox/cuda/tests/test_rng_curand.py",
    "sandbox/cuda/tests/test_nnet.py",
    "sandbox/cuda/tests/test_basic_ops.py",
    "sandbox/cuda/tests/test_memory.py",
    "sandbox/cuda/tests/test_mlp.py",
    "sandbox/cuda/tests/test_bench_loopfusion.py",
    "sandbox/cuda/tests/test_blocksparse.py",
    "sandbox/cuda/tests/test_cuda_ndarray.py",
    "sandbox/cuda/tests/test_tensor_op.py",
    "sandbox/cuda/tests/test_extra_ops.py",
    "sandbox/cuda/tests/test_gemmcorr3d.py",
    "sandbox/cuda/tests/test_viewop.py",
    "sandbox/scan_module/scan_utils.py",
    "sandbox/scan_module/scan.py",
    "sandbox/scan_module/scan_op.py",
    "sandbox/scan_module/__init__.py",
    "sandbox/scan_module/tests/test_utils.py",
    "sandbox/scan_module/tests/test_scan.py",
    "sandbox/linalg/ops.py",
    "sandbox/linalg/kron.py",
    "sandbox/linalg/__init__.py",
    "sandbox/linalg/tests/test_linalg.py",
    "sandbox/gpuarray/comp.py",
    "sandbox/gpuarray/basic_ops.py",
    "sandbox/gpuarray/nnet.py",
    "sandbox/gpuarray/subtensor.py",
    "sandbox/gpuarray/elemwise.py",
    "sandbox/gpuarray/type.py",
    "sandbox/gpuarray/__init__.py",
    "sandbox/gpuarray/opt.py",
    "sandbox/gpuarray/blas.py",
    "sandbox/gpuarray/kernel_codegen.py",
    "sandbox/gpuarray/conv.py",
    "sandbox/gpuarray/neighbours.py",
    "sandbox/gpuarray/tests/test_subtensor.py",
    "sandbox/gpuarray/tests/test_scan.py",
    "sandbox/gpuarray/tests/test_neighbours.py",
    "sandbox/gpuarray/tests/test_conv_cuda_ndarray.py",
    "sandbox/gpuarray/tests/test_type.py",
    "sandbox/gpuarray/tests/test_opt.py",
    "sandbox/gpuarray/tests/test_blas.py",
    "sandbox/gpuarray/tests/test_elemwise.py",
    "sandbox/gpuarray/tests/test_nnet.py",
    "sandbox/gpuarray/tests/test_basic_ops.py",
    "scan_module/scan_utils.py",
    "scan_module/scan_views.py",
    "scan_module/scan.py",
    "scan_module/scan_op.py",
    "scan_module/scan_perform_ext.py",
    "scan_module/__init__.py",
    "scan_module/scan_opt.py",
    "scan_module/tests/test_scan.py",
    "scan_module/tests/test_scan_opt.py",
    "misc/elemwise_openmp_speedup.py",
    "misc/gh_api.py",
    "misc/check_blas.py",
    "misc/latence_gpu_transfert.py",
    "misc/cudamat_utils.py",
    "misc/pycuda_utils.py",
    "misc/pycuda_example.py",
    "misc/ordered_set.py",
    "misc/windows.py",
    "misc/strutil.py",
    "misc/gnumpy_utils.py",
    "misc/may_share_memory.py",
    "misc/safe_asarray.py",
    "misc/pycuda_init.py",
    "misc/check_duplicate_key.py",
    "misc/tests/test_may_share_memory.py",
    "misc/tests/test_pycuda_theano_simple.py",
    "misc/tests/test_gnumpy_utils.py",
    "misc/tests/test_pycuda_utils.py",
    "misc/tests/test_cudamat_utils.py",
    "misc/tests/test_pycuda_example.py",
    "misc/hooks/reindent.py",
    "misc/hooks/check_whitespace.py",
    "sparse/sharedvar.py",
    "sparse/type.py",
    "sparse/__init__.py",
    "sparse/opt.py",
    "sparse/basic.py",
    "sparse/tests/test_utils.py",
    "sparse/tests/test_opt.py",
    "sparse/tests/test_basic.py",
    "sparse/tests/test_sp2.py",
    "sparse/sandbox/test_sp.py",
    "sparse/sandbox/sp2.py",
    "sparse/sandbox/truedot.py",
    "sparse/sandbox/sp.py",
    "gof/destroyhandler.py",
    "gof/vm.py",
    "gof/cutils.py",
    "gof/compiledir.py",
    "gof/unify.py",
    "gof/lazylinker_c.py",
    "gof/optdb.py",
    "gof/utils.py",
    "gof/graph.py",
    "gof/callcache.py",
    "gof/python25.py",
    "gof/type.py",
    "gof/__init__.py",
    "gof/cc.py",
    "gof/opt.py",
    "gof/compilelock.py",
    "gof/link.py",
    "gof/sched.py",
    "gof/toolbox.py",
    "gof/fg.py",
    "gof/op.py",
    "gof/cmodule.py",
    "gof/tests/test_cmodule.py",
    "gof/tests/test_destroyhandler.py",
    "gof/tests/test_opt.py",
    "gof/tests/test_lazy.py",
    "gof/tests/test_toolbox.py",
    "gof/tests/test_link.py",
    "gof/tests/test_fg.py",
    "gof/tests/test_sched.py",
    "gof/tests/test_graph_opt_caching.py",
    "gof/tests/test_graph.py",
    "gof/tests/test_cc.py",
    "gof/tests/test_compute_test_value.py",
    "gof/sandbox/equilibrium.py",
    "sandbox/cuda/opt_util.py",
    "gof/tests/test_utils.py",
    "tensor/tests/_test_mpi_roundtrip.py",
]


def list_files(dir_path=theano.__path__[0], pattern='*.py'):
    """
    List all files under theano's path.
    """
    files_list = []
    for (dir, _, files) in os.walk(dir_path):
        for f in files:
            if fnmatch(f, pattern):
                path = os.path.join(dir, f)
                files_list.append(path)
    return files_list


def test_format_flake8():
    """
    Test if flake8 is respected.
    """
    if not flake8_available:
        raise SkipTest("flake8 is not installed")
    total_errors = 0
    for path in list_files():
        rel_path = os.path.relpath(path, theano.__path__[0])
        if rel_path in whitelist_flake8:
            continue
        else:
            error_num = flake8.main.check_file(path)
            total_errors += error_num
    if total_errors > 0:
        raise AssertionError("FLAKE8 Format not respected")


def print_files_information_flake8():
    """
    Print the list of files which can be removed from the whitelist and the
    list of files which do not respect FLAKE8 formatting that aren't in the
    whitelist.
    """
    infracting_files = []
    non_infracting_files = []
    for path in list_files():
        rel_path = os.path.relpath(path, theano.__path__[0])
        number_of_infractions = flake8.main.check_file(path)
        if number_of_infractions > 0:
            if rel_path not in whitelist_flake8:
                infracting_files.append(rel_path)
        else:
            if rel_path in whitelist_flake8:
                non_infracting_files.append(rel_path)
    print("Files that must be corrected or added to whitelist:")
    for file in infracting_files:
        print(file)
    print("Files that can be removed from whitelist:")
    for file in non_infracting_files:
        print(file)


def check_all_files(dir_path=theano.__path__[0], pattern='*.py'):
    """
    List all .py files under dir_path (theano path), check if they follow
    flake8 format, save all the error-formatted files into
    theano_filelist.txt. This function is used for generating
    the "whitelist_flake8" in this file.
    """

    f_txt = open('theano_filelist.txt', 'a')
    for (dir, _, files) in os.walk(dir_path):
        for f in files:
            if fnmatch(f, pattern):
                error_num = flake8.main.check_file(os.path.join(dir, f))
                if error_num > 0:
                    path = os.path.relpath(os.path.join(dir, f),
                                           theano.__path__[0])
                    f_txt.write('"' + path + '",\n')
    f_txt.close()


if __name__ == "__main__":
    print_files_information_flake8()
