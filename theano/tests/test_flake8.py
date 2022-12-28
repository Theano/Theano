"""
Test flake8 errors.
"""
from __future__ import absolute_import, print_function, division
from nose.plugins.skip import SkipTest
import os
import sys
from fnmatch import fnmatch
import theano
new_flake8 = True
try:
    import flake8.main
    flake8_available = True
    try:
        import flake8.engine
        new_flake8 = False
    except ImportError:
        import flake8.api.legacy
except ImportError:
    flake8_available = False

__authors__ = ("Saizheng Zhang")
__copyright__ = "(c) 2016, Universite de Montreal"
__contact__ = "Saizheng Zhang <saizhenglisa..at..gmail.com>"

# We ignore:
# - "line too long"
#    too complex to do with the C code
# - "closing bracket does not match indentation of opening bracket's line"
#    ignored by default by pep8
# - "'with_statement' missing"
# - "'unicode_literals' missing"
# - "'generator_stop' missing"
# - "'division' present"
# - "'absolute_import' present"
# - "'print_function' present"
# - "expected 2 blank lines after class or function definition"' (E305)
# - "ambiguous variable name" (E741)
#   Redundant error code generated by flake8-future-import module
ignore = ('E501', 'E123', 'E133', 'FI12', 'FI14', 'FI15', 'FI16', 'FI17',
          'FI18', 'FI50', 'FI51', 'FI53', 'E305', 'E741')

whitelist_flake8 = [
    "__init__.py",
    "_version.py",  # This is generated by versioneer
    "tests/__init__.py",
    "compile/__init__.py",
    "compile/sandbox/__init__.py",
    "compile/tests/__init__.py",
    "gpuarray/__init__.py",
    "gpuarray/tests/__init__.py",
    "typed_list/__init__.py",
    "typed_list/tests/__init__.py",
    "tensor/__init__.py",
    "tensor/tests/__init__.py",
    "tensor/tests/test_utils.py",
    "tensor/tests/test_nlinalg.py",
    "tensor/tests/test_shared_randomstreams.py",
    "tensor/tests/test_misc.py",
    "tensor/tests/mlp_test.py",
    "tensor/tests/test_opt_uncanonicalize.py",
    "tensor/tests/test_merge.py",
    "tensor/tests/test_gc.py",
    "tensor/tests/test_complex.py",
    "tensor/tests/test_io.py",
    "tensor/tests/test_sharedvar.py",
    "tensor/tests/test_fourier.py",
    "tensor/tests/test_casting.py",
    "tensor/tests/test_sort.py",
    "tensor/tests/test_raw_random.py",
    "tensor/tests/test_xlogx.py",
    "tensor/tests/test_slinalg.py",
    "tensor/tests/test_blas_c.py",
    "tensor/tests/test_blas_scipy.py",
    "tensor/tests/test_mpi.py",
    "tensor/nnet/__init__.py",
    "tensor/signal/__init__.py",
    "tensor/signal/tests/__init__.py",
    "scalar/__init__.py",
    "scalar/tests/__init__.py",
    "sandbox/__init__.py",
    "sandbox/cuda/__init__.py",
    "sandbox/tests/__init__.py",
    "sandbox/gpuarray/__init__.py",
    "sandbox/linalg/__init__.py",
    "sandbox/linalg/tests/__init__.py",
    "scan_module/scan_utils.py",
    "scan_module/scan_views.py",
    "scan_module/scan.py",
    "scan_module/scan_perform_ext.py",
    "scan_module/__init__.py",
    "scan_module/tests/__init__.py",
    "scan_module/tests/test_scan.py",
    "scan_module/tests/test_scan_opt.py",
    "misc/__init__.py",
    "misc/tests/__init__.py",
    "misc/hooks/reindent.py",
    "misc/hooks/check_whitespace.py",
    "sparse/__init__.py",
    "sparse/tests/__init__.py",
    "sparse/tests/test_utils.py",
    "sparse/tests/test_opt.py",
    "sparse/tests/test_basic.py",
    "sparse/tests/test_sp2.py",
    "sparse/sandbox/__init__.py",
    "sparse/sandbox/test_sp.py",
    "sparse/sandbox/sp2.py",
    "sparse/sandbox/truedot.py",
    "sparse/sandbox/sp.py",
    "gof/__init__.py",
    "d3viz/__init__.py",
    "d3viz/tests/__init__.py",
    "gof/tests/__init__.py",
    "contrib/__init__.py",
]


def list_files(dir_path=theano.__path__[0], pattern='*.py', no_match=".#"):
    """
    List all files under theano's path.
    """
    files_list = []
    for (dir, _, files) in os.walk(dir_path):
        for f in files:
            if fnmatch(f, pattern):
                path = os.path.join(dir, f)
                if not f.startswith(no_match):
                    files_list.append(path)
    return files_list


def test_format_flake8():
    # Test if flake8 is respected.
    if not flake8_available:
        raise SkipTest("flake8 is not installed")
    total_errors = 0

    files_to_checks = []
    for path in list_files():
        rel_path = os.path.relpath(path, theano.__path__[0])
        if sys.platform == 'win32':
            rel_path = rel_path.replace('\\', '/')
        if rel_path in whitelist_flake8:
            continue
        else:
            files_to_checks.append(path)

    if new_flake8:
        guide = flake8.api.legacy.get_style_guide(ignore=ignore)
        r = guide.check_files(files_to_checks)
        total_errors = r.total_errors
    else:
        for path in files_to_checks:
            error_num = flake8.main.check_file(path, ignore=ignore)
            total_errors += error_num

    if total_errors > 0:
        raise AssertionError("FLAKE8 Format not respected")


def print_files_information_flake8(files):
    """
    Print the list of files which can be removed from the whitelist and the
    list of files which do not respect FLAKE8 formatting that aren't in the
    whitelist.
    """
    infracting_files = []
    non_infracting_files = []
    if not files:
        files = list_files()
    for path in files:
        rel_path = os.path.relpath(path, theano.__path__[0])
        number_of_infractions = flake8.main.check_file(path,
                                                       ignore=ignore)
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

    with open('theano_filelist.txt', 'a') as f_txt:
        for (dir, _, files) in os.walk(dir_path):
            for f in files:
                if fnmatch(f, pattern):
                    error_num = flake8.main.check_file(os.path.join(dir, f),
                                                       ignore=ignore)
                    if error_num > 0:
                        path = os.path.relpath(os.path.join(dir, f),
                                               theano.__path__[0])
                        f_txt.write('"' + path + '",\n')


if __name__ == "__main__":
    print_files_information_flake8(sys.argv[1:])
