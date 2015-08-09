import os
import subprocess

def __check_params( params, forbidden = [ 'stdout', 'stderr', 'stdin', ] ) :
    if any( [ par in params for par in forbidden ] ) :
        raise TypeError(
            "Please, do not use the following parameters with either "
            "call_subprocess_Popen() or output_subprocess_Popen(): " +
            ", ".join( forbidden ) )

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

    # Using the dummy file descriptors below is a workaround for a
    # crash experienced in an unusual Python 2.4.4 Windows environment
    # with the default None values.
    stdin = None
    if "stdin" not in params:
        stdin = open( os.devnull, 'r' )
        params['stdin'] = stdin.fileno()

    try:
        proc = subprocess.Popen(command, startupinfo=startupinfo, **params)
    finally:
        if stdin is not None:
            del stdin
    return proc

def call_subprocess_Popen(command, **params):
    """
    Calls subprocess_Popen, discards the output and returns only the exit code.
    :see: documentation for subprocess.Popen for the list of possible parameters.
    """
    __check_params( params )
    with open( os.devnull, 'w' ) as null :
        params['stdout'] = null.fileno( )
        params['stderr'] = null.fileno( )
        proc = subprocess_Popen( command, **params )
        _exit_code = proc.wait( )
    return _exit_code

def output_subprocess_Popen(command, **params):
    """
    Calls subprocess_Popen, returning the output, error and exit code in a tuple.
    :see: documentation for subprocess.Popen for the list of possible parameters.
    """
    __check_params( params )
    params['stdout'] = subprocess.PIPE
    params['stderr'] = subprocess.PIPE
    # Communication with subprocesses should be done with proc.communicate()
    # to avoid deadlocks around the stdour/stderr pipe.
    proc = subprocess_Popen( command, **params )
    _stdout, _stderr = proc.communicate( )
    _exit_code = proc.returncode
    return _stdout, _stderr, _exit_code
