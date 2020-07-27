#!/usr/bin/env python
"""
This script should behave the same as the `nosetests` command.

The reason for its existence is that on some systems, it may not be obvious to
find where nosetests is installed in order to run it in a different process.

It is also used to load the KnownFailure plugin, in order to hide
KnownFailureTests error messages. Use --without-knownfailure to
disable that plugin.

`run_tests_in_batch.py` will in turn call back this script in another process.
"""
from __future__ import print_function

__authors__ = "Olivier Delalleau, Pascal Lamblin, Eric Larsen"
__contact__ = "delallea@iro"

import logging
_logger = logging.getLogger('theano.bin.theano-nose')

import os
import nose
import textwrap
import sys
import warnings
from nose.plugins import Plugin

def main_function():
    # Handle the --theano arguments
    if "--theano" in sys.argv:
        i = sys.argv.index("--theano")
        import theano
        sys.argv[i] = theano.__path__[0]

    # Many Theano tests suppose device=cpu, so we need to raise an
    # error if device==gpu.
    # I don't know how to do this check only if we use theano-nose on
    # Theano tests.  So I make an try..except in case the script get
    # reused elsewhere.
    # We should not import theano before call nose.main()
    # As this cause import problem with nosetests.
    # Should we find a way to don't modify sys.path?
    if not os.path.exists('theano/__init__.py'):
        try:
            from theano import config
            if config.device != "cpu":
                raise ValueError("Theano tests must be run with device=cpu."
                                 " This will also run GPU tests when possible.\n"
                                 " If you want GPU-related tests to run on a"
                                 " specific GPU device, and not the default one,"
                                 " you should use the init_gpu_device theano flag.")
        except ImportError:
            pass

    # Handle --batch[=n] arguments
    batch_args = [arg for arg in sys.argv if arg.startswith('--batch')]
    for arg in batch_args:
        sys.argv.remove(arg)
    batch_size = None
    if len(batch_args):
        if len(batch_args) > 1:
            _logger.warning(
                'Multiple --batch arguments detected, using the last one '
                'and ignoring the first ones.')

        batch_arg = batch_args[-1]
        elems = batch_arg.split('=', 1)
        if len(elems) == 2:
            batch_size = int(elems[1])

    # Handle the --debug-batch argument.
    display_batch_output = False
    if '--debug-batch' in sys.argv:
        if not batch_args:
            raise AssertionError(
                'You can only use the --debug-batch argument with the '
                '--batch[=n] option')
        while '--debug-batch' in sys.argv:
            sys.argv.remove('--debug-batch')
        sys.argv += ['--verbose', '--nocapture', '--detailed-errors']
        display_batch_output = True

    # Handle --time_prof arguments
    time_prof_args = [arg for arg in sys.argv if arg=='--time-profile']
    for arg in time_prof_args:
        sys.argv.remove(arg)

    # Time-profiling and batch modes
    if time_prof_args or batch_args:
        from theano.tests import run_tests_in_batch
        return run_tests_in_batch.main(
                theano_nose=os.path.realpath(__file__),
                batch_size=batch_size,
                time_profile=bool(time_prof_args),
                display_batch_output=display_batch_output)

    # Non-batch mode.
    addplugins = []
    # We include KnownFailure plugin by default, unless
    # it is disabled by the "--without-knownfailure" arg.
    if '--without-knownfailure' not in sys.argv:
        try:
            from numpy.testing.noseclasses import KnownFailure
            addplugins.append(KnownFailure())
        except ImportError:
            _logger.warning(
                'KnownFailure plugin from NumPy could not be imported. '
                'Use --without-knownfailure to disable this warning.')
    else:
        sys.argv.remove('--without-knownfailure')

    # When 'theano-nose' is called-back under the time-profile option, an
    # instance of the custom Nosetests plugin class 'DisabDocString' (see
    # below) is loaded. The latter ensures that the test name will not be
    # replaced in display by the first line of the documentation string.
    if '--disabdocstring' in sys.argv:
        addplugins.append(DisabDocString())

    try:
        if addplugins:
            ret = nose.main(addplugins=addplugins)
        else:
            ret = nose.main()
        return ret
    except TypeError as e:
        if "got an unexpected keyword argument 'addplugins'" in e.message:
            # This means nose is too old and does not support plugins
            _logger.warning(
                'KnownFailure plugin from NumPy can\'t'
                ' be used as nosetests is too old. '
                'Use --without-knownfailure to disable this warning.')
            nose.main()
        else:
            raise


def help():
    help_msg = """
        This script behaves mostly the same as the `nosetests` command.

        The main difference is that it loads automatically the
        KnownFailure plugin, in order to hide KnownFailureTests error
        messages. It also supports executing tests by batches.

        Local options:

            --help, -h: Displays this help.

            --batch[=n]:
                If specified without option '--time-profile', do not run all
                the tests in one run, but split the execution in batches of
                `n` tests each. Default n is 100.

            --time-profile:
                Each test will be run and timed separately and the results will
                be deposited in the files 'timeprof_sort', 'timeprof_nosort'
                and 'timeprof_rawlog' in the current directory. If the
                '--batch[=n]' option is also specified, notification of the
                progresses will be made to standard output after every group of
                n tests. Otherwise, notification will occur after every group
                of 100 tests.

                The files 'timeprof_sort' and 'timeprof_nosort' both contain one
                record for each test and comprise the following fields:
                - test running-time
                - nosetests sequential test number
                - test name
                - name of class to which test belongs (if any), otherwise full
                  information is contained in test name
                - test outcome ('OK', 'SKIPPED TEST', 'FAILED TEST' or
                  'FAILED PARSING')

                In 'timeprof_sort', test records are sorted according to
                running-time whereas in 'timeprof_nosort' records are reported
                according to sequential number. The former classification is the
                main information source for time-profiling. Since tests belonging
                to same or close classes and files have close sequential, the
                latter may be used to identify duration patterns among the tests
                numbers. A full log is also saved as 'timeprof_rawlog'.

            --without-knownfailure: Do not load the KnownFailure plugin.

            --theano: This parameter is replaced with the path to the theano
                      library. As theano-nose is a wrapper to nosetests, it
                      expects a path to the tests to run.
                      If you do not know where theano is installed, use this
                      option to have it inserted automatically.

            --debug-batch:
                Use this parameter to run nosetests with options '--verbose',
                '--nocapture' and '--detailed-errors' and show the output of
                nosetests during batch execution.  This can be useful to debug
                situations where re-running only the failed tests after batch
                execution is not working properly. This option can only be used
                in conjunction with the '--batch=[n]' argument.

        The other options will be passed to nosetests, see ``nosetests -h``.
        """

    print(textwrap.dedent(help_msg))


def main():
    if '--help' in sys.argv or '-h' in sys.argv:
        help()
    else:
        warnings.simplefilter("default")  # Enable warnings before importing theano
        if os.environ.get("PYTHONWARNINGS") is not None:
            os.environ["PYTHONWARNINGS"] = "default"  # Also affect subprocesses
        result = main_function()
        sys.exit(result)


class DisabDocString(Plugin):

    """
    When activated, a custom Nosetests plugin created through this class
    will preclude automatic replacement in display of the name of the test
    by the first line in its documentation string.

    Sources:
    http://nose.readthedocs.org/en/latest/developing.html
    http://nose.readthedocs.org/en/latest/further_reading.html
    http://www.siafoo.net/article/54
    https://github.com/nose-devs/nose/issues/294
    http://python-nose.googlecode.com/svn/trunk/nose/plugins/base.py
    Nat Williams:
    https://github.com/Merino/nose-description-fixer-plugin/commit/
        df94596f29c04fea8001713dd9b04bf3720aebf4
    """

    enabled = False # plugin disabled by default
    score = 2000  # high score ensures priority over other plugins

    def __init__(self):
        # 'super.__init__(self):' would have achieved exactly the same
        if self.name is None:
            self.name = self.__class__.__name__.lower()
        if self.enableOpt is None:
            self.enableOpt = ("enable_plugin_%s"
                              % self.name.replace('-', '_'))

    def options(self, parser, env):
        env_opt = 'NOSE_WITH_%s' % self.name.upper()
        # latter expression to be used if plugin called from the command line
        parser.add_option("--%s" % self.name,
                          # will be called with Nosetests 'main' or 'run'
                          # function's' argument '--disabdocstring'
                          action="store_true",
                          dest=self.enableOpt,
                          # the latter entails that the boolean self.enableOpt
                          # is set to 'True' when plugin is called through a
                          # function's argument
                          default=env.get(env_opt),
                          # entails that plugin will be enabled when command
                          # line trigger 'env_opt' will be activated
                          help="Enable plugin %s: %s [%s]" %
                          (self.__class__.__name__,
                           self.help(), env_opt))

    def configure(self, options, conf):
        self.conf = conf
        # plugin will be enabled when called through argument
        self.enabled = getattr(options, self.enableOpt)

    def describeTest(self, test):
        # 'describeTest' is also called when the test result in Nosetests calls
        # 'test.shortDescription()' and can thus be used to alter the display.
        return False

if __name__ == '__main__':
    main()
