#!/usr/bin/env python


__authors__   = "Olivier Delalleau"
__contact__   = "delallea@iro"


"""
Run this script to run tests in small batches rather than all at the same time.

If no argument is provided, then the whole Theano test-suite is run.
Otherwise, only tests found in the directory given as argument are run.

This script performs three tasks:
    1. Run `nosetests --collect-only --with-id` to collect test IDs
    2. Run `nosetests --with-id i1 ... iN` with batches of N indices, until all
       tests have been run (currently N=100).
    3. Run `nosetests --failed` to re-run only tests that failed
        => The output of this 3rd step is the one you should care about

One reason to use this script is if you are a Windows user, and see errors like
"Not enough storage is available to process this command" when trying to simply
run `nosetests` in your Theano installation directory. This error is apparently
caused by memory fragmentation: at some point Windows runs out of contiguous
memory to load the C modules compiled by Theano in the test-suite.

By using this script, nosetests is run on a small subset (batch) of tests until
all tests are run. Note that this is slower, in particular because of the
initial cost of importing theano and loading the C module cache on each call of
nosetests.
"""


import cPickle, os, subprocess, sys

import theano


def main():
    if len(sys.argv) == 1:
        tests_dir = os.path.join(os.path.dirname(theano.__file__), '..')
    else:
        assert len(sys.argv) == 2
        tests_dir = sys.argv[1]
        assert os.path.isdir(tests_dir)
    os.chdir(tests_dir)
    # It seems safer to fully regenerate the list of tests on each call.
    if os.path.isfile('.noseids'):
        os.remove('.noseids')
    # Collect test IDs.
    print """\
####################
# COLLECTING TESTS #
####################"""
    assert subprocess.call(['nosetests', '--collect-only', '--with-id']) == 0
    noseids_file = os.path.join(tests_dir, '.noseids')
    data = cPickle.load(open(noseids_file, 'rb'))
    ids = data['ids']
    n_tests = len(ids)
    assert n_tests == max(ids)
    # Run tests.
    n_batch = 10
    failed = []
    print """\
###################################
# RUNNING TESTS IN BATCHES OF %s #
###################################""" % n_batch
    for test_id in xrange(1, n_tests + 1, n_batch):
        test_range = range(test_id, min(test_id + n_batch, n_tests + 1))
        # We suppress all output because we want the user to focus only on the
        # failed tests, which are re-run (with output) below.
        dummy_out = open(os.devnull, 'w')
        rval = subprocess.call(['nosetests', '-q', '--with-id'] +
                               map(str, test_range), stdout=dummy_out.fileno(),
                               stderr=dummy_out.fileno())
        # Recover failed test indices from the 'failed' field of the '.noseids'
        # file. We need to do it after each batch because otherwise this field
        # gets erased.
        failed += cPickle.load(open(noseids_file, 'rb'))['failed']
        print '%s%% done (failed: %s)' % ((test_range[-1] * 100) // n_tests,
                                          len(failed))
    if failed:
        # Re-run only failed tests
        print """\
################################
# RE-RUNNING FAILED TESTS ONLY #
################################"""
        subprocess.call(['nosetests', '-v', '--with-id'] + failed)
        return 0
    else:
        print """\
####################
# ALL TESTS PASSED #
####################"""


if __name__ == '__main__':
    sys.exit(main())
