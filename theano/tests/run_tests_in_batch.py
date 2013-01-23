#!/usr/bin/env python

__authors__ = "Olivier Delalleau, Eric Larsen"
__contact__ = "delallea@iro"

"""
Run this script to run tests in small batches rather than all at the same time
or to conduct time-profiling.

If no argument is provided, then the whole Theano test-suite is run.
Otherwise, only tests found in the directory given as argument are run.

If 'time_profile=False', this script performs three tasks:
    1. Run `nosetests --collect-only --with-id` to collect test IDs
    2. Run `nosetests --with-id i1 ... iN` with batches of 'batch_size'
       indices, until all tests have been run (currently batch_size=100 by
       default).
    3. Run `nosetests --failed` to re-run only tests that failed
       => The output of this 3rd step is the one you should care about

If 'time_profile=True', this script conducts time-profiling of the tests:
    1. Run `nosetests --collect-only --with-id` to collect test IDs
    2. Run `nosetests --with-id i`, one test with ID 'i' at a time, collecting
       timing information and displaying progresses on standard output after
       every group of 'batch_size' (100 by default), until all tests have
       been run.
       The results are deposited in the files 'timeprof_sort' and
       'timeprof_nosort' in the current directory. Both contain one record for
       each test and comprise the following fields:
       - test running-time
       - nosetests sequential test number
       - test name
       - name of class to which test belongs (if any), otherwise full
         information is contained in test name
       - test outcome ('OK', 'SKIPPED TEST', 'FAILED TEST' or 'FAILED PARSING')
       In 'timeprof_sort', test records are sorted according to run-time
       whereas in 'timeprof_nosort' records are reported according to
       sequential number. The former classification is the main information
       source for time-profiling. Since tests belonging to same or close
       classes and files have close sequential numbers, the latter may be used
       to identify duration patterns among the tests. A full log is also saved
       as 'timeprof_rawlog'.

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


import cPickle
import os
import subprocess
import sys
import datetime
import theano
from theano.misc.windows import call_subprocess_Popen


def main(stdout=None, stderr=None, argv=None, theano_nose=None,
         batch_size=None, time_profile=False, display_batch_output=False):
    """
    Run tests with optional output redirection.

    Parameters stdout and stderr should be file-like objects used to redirect
    the output. None uses default sys.stdout and sys.stderr.

    If argv is None, then we use arguments from sys.argv, otherwise we use the
    provided arguments instead.

    If theano_nose is None, then we use the theano-nose script found in
    Theano/bin to call nosetests. Otherwise we call the provided script.

    If batch_size is None, we use a default value of 100.

    If display_batch_output is False, then the output of nosetests during batch
    execution is hidden.
    """

    if stdout is None:
        stdout = sys.stdout
    if stderr is None:
        stderr = sys.stderr
    if argv is None:
        argv = sys.argv
    if theano_nose is None:
    #If Theano is installed with pip/easy_install, it can be in the
    #*/lib/python2.7/site-packages/theano, but theano-nose in */bin
        for i in range(1, 5):
            path = theano.__path__[0]
            for _ in range(i):
                path = os.path.join(path, '..')
            path = os.path.join(path, 'bin', 'theano-nose')
            if os.path.exists(path):
                theano_nose = path
                break
    if theano_nose is None:
        raise Exception("Not able to find theano_nose")
    if batch_size is None:
        batch_size = 100
    stdout_backup = sys.stdout
    stderr_backup = sys.stderr
    try:
        sys.stdout = stdout
        sys.stderr = stderr
        run(stdout, stderr, argv, theano_nose, batch_size, time_profile,
            display_batch_output)
    finally:
        sys.stdout = stdout_backup
        sys.stderr = stderr_backup


def run(stdout, stderr, argv, theano_nose, batch_size, time_profile,
        display_batch_output):

    # Setting aside current working directory for later saving
    sav_dir = os.getcwd()
    # The first argument is the called script.
    argv = argv[1:]

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
    # We need to call 'python' on Windows, because theano-nose is not a
    # native Windows app; and it does not hurt to call it on Unix.
    # Using sys.executable, so that the same Python version is used.
    python = sys.executable
    rval = subprocess.call(
        ([python, theano_nose, '--collect-only', '--with-id']
         + argv),
        stdin=dummy_in.fileno(),
        stdout=stdout.fileno(),
        stderr=stderr.fileno())
    stdout.flush()
    stderr.flush()
    assert rval == 0
    noseids_file = '.noseids'
    data = cPickle.load(open(noseids_file, 'rb'))
    ids = data['ids']
    n_tests = len(ids)
    assert n_tests == max(ids)

    # Standard batch testing is called for
    if not time_profile:
        failed = set()
        print """\
###################################
# RUNNING TESTS IN BATCHES OF %s #
###################################""" % batch_size
        # When `display_batch_output` is False, we suppress all output because
        # we want the user to focus only on the failed tests, which are re-run
        # (with output) below.
        dummy_out = open(os.devnull, 'w')
        for test_id in xrange(1, n_tests + 1, batch_size):
            stdout.flush()
            stderr.flush()
            test_range = range(test_id, min(test_id + batch_size, n_tests + 1))
            cmd = ([python, theano_nose, '--with-id'] +
                   map(str, test_range) +
                   argv)
            subprocess_extra_args = dict(stdin=dummy_in.fileno())
            if not display_batch_output:
                # Use quiet mode in nosetests.
                cmd.append('-q')
                # Suppress all output.
                subprocess_extra_args.update(dict(
                    stdout=dummy_out.fileno(),
                    stderr=dummy_out.fileno()))
            subprocess.call(cmd, **subprocess_extra_args)
            # Recover failed test indices from the 'failed' field of the
            # '.noseids' file. We need to do it after each batch because
            # otherwise this field may get erased. We use a set because it
            # seems like it is not systematically erased though, and we want
            # to avoid duplicates.
            failed = failed.union(cPickle.load(open(noseids_file, 'rb'))
                                  ['failed'])
            print '%s%% done (failed: %s)' % ((test_range[-1] * 100) //
                                n_tests, len(failed))
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
            subprocess.call(
                ([python, theano_nose, '-v', '--with-id']
                 + failed
                 + argv),
                stdin=dummy_in.fileno(),
                stdout=stdout.fileno(),
                stderr=stderr.fileno())
            stdout.flush()
            stderr.flush()
            return 0
        else:
            print """\
####################
# ALL TESTS PASSED #
####################"""

    # Time-profiling is called for
    else:
        print """\
########################################
# RUNNING TESTS IN TIME-PROFILING MODE #
########################################"""

        # finds first word of list l containing string s
        def getIndexOfFirst(l, s):
            for pos, word in enumerate(l):
                if s in word:
                    return pos

        # finds last word of list l containing string s
        def getIndexOfLast(l, s):
            for pos, word in enumerate(reversed(l)):
                if s in word:
                    return (len(l) - pos - 1)

        # iterating through tests
        # initializing master profiling list and raw log
        prof_master_nosort = []
        prof_rawlog = []
        dummy_out = open(os.devnull, 'w')
        path_rawlog = os.path.join(sav_dir, 'timeprof_rawlog')
        stamp = str(datetime.datetime.now()) + '\n\n'
        f_rawlog = open(path_rawlog, 'w')
        f_rawlog.write('TIME-PROFILING OF THEANO\'S NOSETESTS'
                       ' (raw log)\n\n' + stamp)
        f_rawlog.flush()

        stamp = str(datetime.datetime.now()) + '\n\n'
        fields = ('Fields: computation time; nosetests sequential id;'
                  ' test name; parent class (if any); outcome\n\n')
        path_nosort = os.path.join(sav_dir, 'timeprof_nosort')
        f_nosort = open(path_nosort, 'w')
        f_nosort.write('TIME-PROFILING OF THEANO\'S NOSETESTS'
                       ' (by sequential id)\n\n' + stamp + fields)
        f_nosort.flush()
        for test_floor in xrange(1, n_tests + 1, batch_size):
            for test_id in xrange(test_floor, min(test_floor + batch_size,
                                                 n_tests + 1)):
                # Print the test we will start in the raw log to help
                # debug tests that are too long.
                f_rawlog.write("\nWill run test #%d %s\n" % (test_id,
                                                         data["ids"][test_id]))
                f_rawlog.flush()

                proc = call_subprocess_Popen(
                    ([python, theano_nose, '-v', '--with-id']
                    + [str(test_id)] + argv +
                     ['--disabdocstring']),
                    # the previous option calls a custom Nosetests plugin
                    # precluding automatic sustitution of doc. string for
                    # test name in display
                    # (see class 'DisabDocString' in file theano-nose)
                    stderr=subprocess.PIPE,
                    stdout=dummy_out.fileno(),
                    stdin=dummy_in.fileno())

                # recovering and processing data from pipe
                err = proc.stderr.read()
                # print the raw log
                f_rawlog.write(err)
                f_rawlog.flush()

                # parsing the output
                l_err = err.split()
                try:
                    pos_id = getIndexOfFirst(l_err, '#')
                    prof_id = l_err[pos_id]
                    pos_dot = getIndexOfFirst(l_err, '...')
                    prof_test = ''
                    for s in l_err[pos_id + 1: pos_dot]:
                        prof_test += s + ' '
                    if 'OK' in err:
                        pos_ok = getIndexOfLast(l_err, 'OK')
                        if len(l_err) == pos_ok + 1:
                            prof_time = float(l_err[pos_ok - 1][0:-1])
                            prof_pass = 'OK'
                        elif 'SKIP' in l_err[pos_ok + 1]:
                            prof_time = 0.
                            prof_pass = 'SKIPPED TEST'
                        elif 'KNOWNFAIL' in l_err[pos_ok + 1]:
                            prof_time = float(l_err[pos_ok - 1][0:-1])
                            prof_pass = 'OK'
                        else:
                            prof_time = 0.
                            prof_pass = 'FAILED TEST'
                    else:
                        prof_time = 0.
                        prof_pass = 'FAILED TEST'
                except Exception:
                    prof_time = 0
                    prof_id = '#' + str(test_id)
                    prof_test = ('FAILED PARSING, see raw log for details'
                                 ' on test')
                    prof_pass = ''
                prof_tuple = (prof_time, prof_id, prof_test, prof_pass)

                # appending tuple to master list
                prof_master_nosort.append(prof_tuple)

                # write the no sort file
                s_nosort = ((str(prof_tuple[0]) + 's').ljust(10) +
                 " " + prof_tuple[1].ljust(7) + " " +
                 prof_tuple[2] + prof_tuple[3] +
                 "\n")
                f_nosort.write(s_nosort)
                f_nosort.flush()

            print '%s%% time-profiled' % ((test_id * 100) // n_tests)
        f_rawlog.close()

        # sorting tests according to running-time
        prof_master_sort = sorted(prof_master_nosort,
                                  key=lambda test: test[0], reverse=True)

        # saving results to readable files
        path_sort = os.path.join(sav_dir, 'timeprof_sort')
        f_sort = open(path_sort, 'w')
        f_sort.write('TIME-PROFILING OF THEANO\'S NOSETESTS'
                     ' (sorted by computation time)\n\n' + stamp + fields)
        for i in xrange(len(prof_master_nosort)):
            s_sort = ((str(prof_master_sort[i][0]) + 's').ljust(10) +
                 " " + prof_master_sort[i][1].ljust(7) + " " +
                 prof_master_sort[i][2] + prof_master_sort[i][3] +
                 "\n")
            f_sort.write(s_sort)
        f_nosort.close()
        f_sort.close()

if __name__ == '__main__':
    sys.exit(main())
