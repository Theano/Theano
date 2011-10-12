__copyright__ = "(c) 2011, Universite de Montreal"
__license__   = "3-clause BSD License"


import warnings

from theano.sandbox.cuda.nvcc_compiler import remove_python_framework_dir


def test_remove_python_framework_dir():
    """
    Test function 'remove_python_framework_dir'.
    """
    # This is a typical output of 'python-config --ldflags'.
    cmd = ('-L/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/config '
           '-ldl -framework CoreFoundation -lpython2.7 -u _PyMac_Error '
           '/opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python').split()
    assert remove_python_framework_dir(cmd) == cmd[0:-1]
    # Add a fake argument that should not be removed.
    cmd.append(
        '-L/opt/local/Library/Frameworks/Python.framework/Versions/2.7/Python')
    assert remove_python_framework_dir(cmd) == cmd[0:-2] + cmd[-1:]
    # We test for the warning only if we can use 'catch_warnings' (Python 2.6+)
    # as otherwise it is difficult to do it properly.
    try:
        warnings.catch_warnings
    except AttributeError:
        return
    cmd.append('Frameworks/Python.framework/Versions/2.6/Python')

    try:
        record = warnings.catch_warnings(record=True)
        record2 = record.__enter__()

        assert remove_python_framework_dir(cmd) == cmd[0:-3] + cmd[-2:-1]
        assert len(record2) == 1
        assert 'remove_python_framework_dir' in str(record2[0].message)
    finally:
        record.__exit__(None, None, None)
