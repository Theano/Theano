import os
import subprocess


def call_subprocess_Popen(command, **params):
    """
    Utility function to work around windows behavior that open windows
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
        stdin = open(os.devnull)
        params['stdin'] = stdin.fileno()

    try:
        proc = subprocess.Popen(command, startupinfo=startupinfo, **params)
    finally:
        if stdin is not None:
            del stdin
    return proc
