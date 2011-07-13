"""
This file is used to check if scipy is available. We do it at only one
place to localise this check. It is to the module to check the scipy
version.
"""

try:
    import scipy
    scipy_available = True
except ImportError:
    scipy_available = False
