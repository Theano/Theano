#!/usr/bin/env python
from __future__ import print_function
import logging
import os
import sys

if sys.platform == 'win32':
    config_for_theano_cache_script = 'cxx=,device=cpu'
    theano_flags = os.environ['THEANO_FLAGS'] if 'THEANO_FLAGS' in os.environ else ''
    if theano_flags:
        theano_flags += ','
    theano_flags += config_for_theano_cache_script
    os.environ['THEANO_FLAGS'] = theano_flags

import theano
from theano import config
import theano.gof.compiledir
from theano.gof.cc import get_module_cache

_logger = logging.getLogger('theano.bin.theano-cache')


def print_help(exit_status):
    if exit_status:
        print('command "%s" not recognized' % (' '.join(sys.argv)))
    print('Type "theano-cache" to print the cache location')
    print('Type "theano-cache help" to print this help')
    print('Type "theano-cache clear" to erase the cache')
    print('Type "theano-cache list" to print the cache content')
    print('Type "theano-cache unlock" to unlock the cache directory')
    print('Type "theano-cache cleanup" to delete keys in the old '
          'format/code version')
    print('Type "theano-cache purge" to force deletion of the cache directory')
    print('Type "theano-cache basecompiledir" '
          'to print the parent of the cache directory')
    print('Type "theano-cache basecompiledir list" '
          'to print the content of the base compile dir')
    print('Type "theano-cache basecompiledir purge" '
          'to remove everything in the base compile dir, '
          'that is, erase ALL cache directories')
    sys.exit(exit_status)


def main():
    if len(sys.argv) == 1:
        print(config.compiledir)
    elif len(sys.argv) == 2:
        if sys.argv[1] == 'help':
            print_help(exit_status=0)
        if sys.argv[1] == 'clear':
            # We skip the refresh on module cache creation because the refresh will
            # be done when calling clear afterwards.
            cache = get_module_cache(init_args=dict(do_refresh=False))
            cache.clear(unversioned_min_age=-1, clear_base_files=True,
                        delete_if_problem=True)

            # Print a warning if some cached modules were not removed, so that the
            # user knows he should manually delete them, or call
            # theano-cache purge, # to properly clear the cache.
            items = [item for item in sorted(os.listdir(cache.dirname))
                     if item.startswith('tmp')]
            if items:
                _logger.warning(
                    'There remain elements in the cache dir that you may '
                    'need to erase manually. The cache dir is:\n  %s\n'
                    'You can also call "theano-cache purge" to '
                    'remove everything from that directory.' %
                    config.compiledir)
                _logger.debug('Remaining elements (%s): %s' %
                              (len(items), ', '.join(items)))
        elif sys.argv[1] == 'list':
            theano.gof.compiledir.print_compiledir_content()
        elif sys.argv[1] == 'cleanup':
            theano.gof.compiledir.cleanup()
            cache = get_module_cache(init_args=dict(do_refresh=False))
            cache.clear_old()
        elif sys.argv[1] == 'unlock':
            theano.gof.compilelock.force_unlock()
            print('Lock successfully removed!')
        elif sys.argv[1] == 'purge':
            theano.gof.compiledir.compiledir_purge()
        elif sys.argv[1] == 'basecompiledir':
            # Simply print the base_compiledir
            print(theano.config.base_compiledir)
        else:
            print_help(exit_status=1)
    elif len(sys.argv) == 3 and sys.argv[1] == 'basecompiledir':
        if sys.argv[2] == 'list':
            theano.gof.compiledir.basecompiledir_ls()
        elif sys.argv[2] == 'purge':
            theano.gof.compiledir.basecompiledir_purge()
        else:
            print_help(exit_status=1)
    else:
        print_help(exit_status=1)


if __name__ == '__main__':
    main()
