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
        test_warning = True
    except AttributeError:
        test_warning = False
    if test_warning:
        cmd.append('Frameworks/Python.framework/Versions/2.6/Python')
        # Python 2.4 "emulation" of `with` statement. It is necessary even if this
        # code is not executed, because using `with` would result in a SyntaxError.
        with_context = warnings.catch_warnings(record=True)
        record = with_context.__enter__()
        try:
            assert remove_python_framework_dir(cmd) == cmd[0:-3] + cmd[-2:-1]
            assert len(record) == 1
            assert 'remove_python_framework_dir' in str(record[0].message)
        finally:
            with_context.__exit__(None, None, None)

    # Now test some more typical arguments that should be caught by the regex.
    for arg_to_remove in [
            '/Library/Frameworks/EPD64.framework/Versions/7.1/Python',
            '/Library/Frameworks/Python.framework/Versions/7.2/Python',
            ]:
        # Make sure those arguments are removed.
        assert not remove_python_framework_dir([arg_to_remove])
