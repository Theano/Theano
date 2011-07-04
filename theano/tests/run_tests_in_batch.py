#!/usr/bin/env python


__authors__   = "Olivier Delalleau"
__contact__   = "delallea@iro"


"""
Run this script to run tests individually.

This script performs three tasks:
    1. Run `nosetests --collect-only --with-id` to collect test IDs
    2. Run `nosetests --with-id X` with for X = 1 to total number of tests
    3. Run `nosetests --failed` to re-run only tests that failed
        => The output of this 3rd step is the one you should care about

One reason to use this script is if you are a Windows user, and see errors like
"Not enough storage is available to process this command" when trying to simply
run `nosetests` in your Theano installation directory.
By using this script, nosetests is run on each test individually, which is much
slower but should at least let you run the test suite.

Note that this script is a work-in-progress and is not fully functional at this
point: the way some tests are defined in the Theano test-suite seems to confuse
the nosetests' TestID module, probably leading to not running all tests, as
well as to some unexpected test failures.

You can also provide a single command-line argument, which should be an integer
number N (default = 1), in order to run batches of N tests rather than run tests
one at a time. It will be faster (but may fail under Windows if N is too large).
"""


import cPickle, os, subprocess, sys

import theano


def main():
    theano_install_dir = os.path.join(os.path.dirname(theano.__file__), '..')
    os.chdir(theano_install_dir)
    # It seems like weird things happen if we keep the same IDs file around...
    # (the number of test items in it changes from one run to another)
    if os.path.isfile('.noseids'):
        os.remove('.noseids')
    # Collect test IDs.
    assert subprocess.call(['nosetests', '--collect-only', '--with-id']) == 0
    data = cPickle.load(
            open(os.path.join(theano_install_dir, '.noseids'), 'rb'))
    ids = data['ids']
    n_tests = len(ids)
    # Run tests.
    n_batch = 1
    if len(sys.argv) >= 2:
        n_batch = int(sys.argv[1])
    has_error = 0
    for test_id in xrange(1, n_tests + 1, n_batch):
        test_range = range(test_id, min(test_id + n_batch, n_tests + 1))
        rval = subprocess.call(['nosetests', '-v', '--with-id'] +
                               map(str, test_range))
        has_error += rval
    if has_error:
        # Re-run only failed tests
        print """\
###########################
# RE-RUNNING FAILED TESTS #
###########################"""
        subprocess.call(['nosetests', '-v', '--failed'])
        return 0
    else:
        print """\
####################
# ALL TESTS PASSED #
####################"""


if __name__ == '__main__':
    sys.exit(main())
