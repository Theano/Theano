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
By using this script, nosetests is run on each test individually, which is
slower but should at least let you run the test suite.

Note that this script is experimental and might not actually be running the
whole test suite (it is suspicious that the first step reports more tests than
the amount of tests found in the resulting '.noseids' file).
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
    for test_id in xrange(1, n_tests + 1):
        subprocess.call(['nosetests', '-v', '--with-id', str(test_id)])
    # Re-run only failed tests
    subprocess.call(['nosetests', '-v', '--failed'])
    return 0


if __name__ == '__main__':
    sys.exit(main())

