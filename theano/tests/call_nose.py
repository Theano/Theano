__authors__   = "Olivier Delalleau"
__contact__   = "delallea@iro"


"""
This script should behave the same as the `nosetests` command.

The reason for its existence is that on some systems, it may not be obvious to
find where nosetests is installed in order to run it in a different process.
This script is called from `run_tests_in_batch.py`.
"""


import sys

import nose


if __name__ == '__main__':
    sys.exit(nose.main())
