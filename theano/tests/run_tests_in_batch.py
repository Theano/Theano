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


def main(stdout=None, stderr=None, argv=None, call_nose=None):
    """
    Run tests with optional output redirection.

    Parameters stdout and stderr should be file-like objects used to redirect
    the output. None uses default sys.stdout and sys.stderr.

    If argv is None, then we use arguments from sys.argv, otherwise we use the
    provided arguments instead.

    If call_nose is None, then we use the call_nose.py script found in
    theano/tests to call nosetests. Otherwise we call the provided script.
    """
    if stdout is None:
        stdout = sys.stdout
    if stderr is None:
        stderr = sys.stderr
    if argv is None:
        argv = sys.argv
    if call_nose is None:
        call_nose = os.path.join(theano.__path__[0], 'tests', 'call_nose.py')
    stdout_backup = sys.stdout
    stderr_backup = sys.stderr
    try:
        sys.stdout = stdout
        sys.stderr = stderr
        run(stdout, stderr, argv, call_nose)
    finally:
        sys.stdout = stdout_backup
        sys.stderr = stderr_backup

def run(stdout, stderr, argv, call_nose):
    if len(argv) == 1:
        tests_dir = theano.__path__[0]
    else:
        assert len(argv) == 2
        tests_dir = argv[1]
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
    stdout.flush()
    stderr.flush()
    dummy_in = open(os.devnull)
    rval = subprocess.call(['python', call_nose, '--collect-only', '--with-id'],
                           stdin=dummy_in.fileno(), stdout=stdout.fileno(),
                           stderr=stderr.fileno())
    stdout.flush()
    stderr.flush()
    assert rval == 0
    noseids_file = '.noseids'
    data = cPickle.load(open(noseids_file, 'rb'))
    ids = data['ids']
    n_tests = len(ids)
    assert n_tests == max(ids)
    # Run tests.
    n_batch = 100
    failed = set()
    print """\
###################################
# RUNNING TESTS IN BATCHES OF %s #
###################################""" % n_batch
    for test_id in xrange(1, n_tests + 1, n_batch):
        stdout.flush()
        stderr.flush()
        test_range = range(test_id, min(test_id + n_batch, n_tests + 1))
        # We suppress all output because we want the user to focus only on the
        # failed tests, which are re-run (with output) below.
        dummy_out = open(os.devnull, 'w')
        rval = subprocess.call(['python', call_nose, '-q', '--with-id'] +
                               map(str, test_range), stdout=dummy_out.fileno(),
                               stderr=dummy_out.fileno(),
                               stdin=dummy_in.fileno())
        # Recover failed test indices from the 'failed' field of the '.noseids'
        # file. We need to do it after each batch because otherwise this field
        # may get erased. We use a set because it seems like it is not
        # systematically erased though, and we want to avoid duplicates.
        failed = failed.union(cPickle.load(open(noseids_file, 'rb'))['failed'])
        print '%s%% done (failed: %s)' % ((test_range[-1] * 100) // n_tests,
                                          len(failed))
    # Sort for cosmetic purpose only.
    failed = sorted(failed)
    if failed:
        # Re-run only failed tests
        print """\
################################
# RE-RUNNING FAILED TESTS ONLY #
################################"""
        stdout.flush()
        stderr.flush()
        subprocess.call(['python', call_nose, '-v', '--with-id'] + failed,
                        stdin=dummy_in.fileno(), stdout=stdout.fileno(),
                        stderr=stderr.fileno())
        stdout.flush()
        stderr.flush()
        return 0
    else:
        print """\
####################
# ALL TESTS PASSED #
####################"""


if __name__ == '__main__':
    sys.exit(main())
