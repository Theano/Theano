from __future__ import absolute_import, print_function, division
import os
import subprocess


def subprocess_Popen(command, **params):
    """
    Utility function to work around windows behavior that open windows.

    :see: call_subprocess_Popen and output_subprocess_Popen
    """
    startupinfo = None
    if os.name == 'nt':
        startupinfo = subprocess.STARTUPINFO()
        try:
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        except AttributeError:
            startupinfo.dwFlags |= subprocess._subprocess.STARTF_USESHOWWINDOW

        # Anaconda for Windows does not always provide .exe files
        # in the PATH, they also have .bat files that call the corresponding
        # executable. For instance, "g++.bat" is in the PATH, not "g++.exe"
        # Unless "shell=True", "g++.bat" is not executed when trying to
        # execute "g++" without extensions.
        # (Executing "g++.bat" explicitly would also work.)
        params['shell'] = True
        # "If shell is True, it is recommended to pass args as a string rather than as a sequence." (cite taken from https://docs.python.org/2/library/subprocess.html#frequently-used-arguments)
        # In case when command arguments have spaces, passing a command as a list will result in incorrect arguments break down, and consequently
        # in "The filename, directory name, or volume label syntax is incorrect" error message.
        # Passing the command as a single string solves this problem.
        if isinstance(command, list):
            command = ' '.join(command)

    # Using the dummy file descriptors below is a workaround for a
    # crash experienced in an unusual Python 2.4.4 Windows environment
    # with the default None values.
    stdin = None
    if "stdin" not in params:
        stdin = open(os.devnull)
        params['stdin'] = stdin.fileno()

    try:
        proc = subprocess.Popen(command, startupinfo=startupinfo, **params)
    finally:
        if stdin is not None:
            stdin.close()
    return proc


def call_subprocess_Popen(command, **params):
    """
    Calls subprocess_Popen and discards the output, returning only the
    exit code.
    """
    if 'stdout' in params or 'stderr' in params:
        raise TypeError("don't use stderr or stdout with call_subprocess_Popen")
    with open(os.devnull, 'wb') as null:
        # stdin to devnull is a workaround for a crash in a weird Windows
        # environment where sys.stdin was None
        params.setdefault('stdin', null)
        params['stdout'] = null
        params['stderr'] = null
        p = subprocess_Popen(command, **params)
        returncode = p.wait()
    return returncode


def output_subprocess_Popen(command, **params):
    """
    Calls subprocess_Popen, returning the output, error and exit code
    in a tuple.
    """
    if 'stdout' in params or 'stderr' in params:
        raise TypeError("don't use stderr or stdout with output_subprocess_Popen")
    params['stdout'] = subprocess.PIPE
    params['stderr'] = subprocess.PIPE
    p = subprocess_Popen(command, **params)
    # we need to use communicate to make sure we don't deadlock around
    # the stdout/stderr pipe.
    out = p.communicate()
    return out + (p.returncode,)
