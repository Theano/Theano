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

        # Under Windows 7 64-bits, Anaconda's g++ is not found unless
        # specifying "shell=True".
        params['shell'] = True
    proc = subprocess.Popen(command, startupinfo=startupinfo, **params)
    return proc
