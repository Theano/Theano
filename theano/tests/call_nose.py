__authors__   = "Olivier Delalleau"
__contact__   = "delallea@iro"


"""
This script should behave the same as the `nosetests` command.

The reason for its existence is that on some systems, it may not be obvious to
find where nosetests is installed in order to run it in a different process.

This script is called from `run_tests_in_batch.py`.

It is also used to load the KnownFailure plugin, in order to hide
KnownFailureTests error messages.
"""

import nose
import sys

if __name__ == '__main__':
    addplugins = []
    # We include KnownFailure plugin by default, unless
    # it is disabled by the "--without-knownfailure" arg.
    if '--without-knownfailure' not in sys.argv:
        try:
            from numpy.testing.noseclasses import KnownFailure
            addplugins.append(KnownFailure())
        except ImportError:
            pass
    else:
        sys.argv.remove('--without-knownfailure')

    sys.exit(nose.main(addplugins=addplugins))
