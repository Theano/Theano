from __future__ import absolute_import, print_function, division
import six.moves.cPickle as pickle
import logging
import os
import shutil

import numpy as np

import theano
from six import string_types, iteritems
from theano import config
from theano.gof.utils import flatten


_logger = logging.getLogger("theano.gof.compiledir")


def cleanup():
    """
    Delete keys in old format from the compiledir.

    Old clean up include key in old format or with old version of the c_code:
    1) keys that have an ndarray in them.
       Now we use a hash in the keys of the constant data.
    2) key that don't have the numpy ABI version in them
    3) They do not have a compile version string

    If there is no key left for a compiled module, we delete the module.

    """
    compiledir = theano.config.compiledir
    for directory in os.listdir(compiledir):
        file = None
        try:
            try:
                filename = os.path.join(compiledir, directory, "key.pkl")
                file = open(filename, 'rb')
                # print file
                try:
                    keydata = pickle.load(file)
                    for key in list(keydata.keys):
                        have_npy_abi_version = False
                        have_c_compiler = False
                        for obj in flatten(key):
                            if isinstance(obj, np.ndarray):
                                # Reuse have_npy_abi_version to
                                # force the removing of key
                                have_npy_abi_version = False
                                break
                            elif isinstance(obj, string_types):
                                if obj.startswith('NPY_ABI_VERSION=0x'):
                                    have_npy_abi_version = True
                                elif obj.startswith('c_compiler_str='):
                                    have_c_compiler = True
                            elif (isinstance(obj, (theano.gof.Op,
                                                   theano.gof.Type)) and
                                  hasattr(obj, 'c_code_cache_version')):
                                v = obj.c_code_cache_version()
                                if v not in [(), None] and v not in key[0]:
                                    # Reuse have_npy_abi_version to
                                    # force the removing of key
                                    have_npy_abi_version = False
                                    break

                        if not have_npy_abi_version or not have_c_compiler:
                            try:
                                # This can happen when we move the compiledir.
                                if keydata.key_pkl != filename:
                                    keydata.key_pkl = filename
                                keydata.remove_key(key)
                            except IOError:
                                _logger.error(
                                    "Could not remove file '%s'. To complete "
                                    "the clean-up, please remove manually "
                                    "the directory containing it.",
                                    filename)
                    if len(keydata.keys) == 0:
                        shutil.rmtree(os.path.join(compiledir, directory))

                except (EOFError, AttributeError):
                    _logger.error(
                        "Could not read key file '%s'. To complete "
                        "the clean-up, please remove manually "
                        "the directory containing it.",
                        filename)
            except IOError:
                _logger.error(
                    "Could not clean up this directory: '%s'. To complete "
                    "the clean-up, please remove it manually.",
                    directory)
        finally:
            if file is not None:
                file.close()


def print_title(title, overline='', underline=''):
    len_title = len(title)
    if overline:
        print(str(overline) * len_title)
    print(title)
    if underline:
        print(str(underline) * len_title)


def print_compiledir_content():
    """
    print list of %d compiled individual ops in the "theano.config.compiledir"
    """
    max_key_file_size = 1 * 1024 * 1024  # 1M

    compiledir = theano.config.compiledir
    table = []
    table_multiple_ops = []
    table_op_class = {}
    zeros_op = 0
    big_key_files = []
    total_key_sizes = 0
    nb_keys = {}
    for dir in os.listdir(compiledir):
        filename = os.path.join(compiledir, dir, "key.pkl")
        if not os.path.exists(filename):
            continue
        with open(filename, 'rb') as file:
            try:
                keydata = pickle.load(file)
                ops = list(set([x for x in flatten(keydata.keys)
                                if isinstance(x, theano.gof.Op)]))
                # Whatever the case, we count compilations for OP classes.
                for op_class in set([op.__class__ for op in ops]):
                    table_op_class.setdefault(op_class, 0)
                    table_op_class[op_class] += 1
                if len(ops) == 0:
                    zeros_op += 1
                else:
                    types = list(set([x for x in flatten(keydata.keys)
                                      if isinstance(x, theano.gof.Type)]))
                    compile_start = compile_end = float('nan')
                    for fn in os.listdir(os.path.join(compiledir, dir)):
                        if fn.startswith('mod.c'):
                            compile_start = os.path.getmtime(
                                os.path.join(compiledir, dir, fn))
                        elif fn.endswith('.so'):
                            compile_end = os.path.getmtime(
                                os.path.join(compiledir, dir, fn))
                    compile_time = compile_end - compile_start
                    if len(ops) == 1:
                        table.append((dir, ops[0], types, compile_time))
                    else:
                        ops_to_str = '[%s]' % ', '.join(sorted(str(op) for op in ops))
                        types_to_str = '[%s]' % ', '.join(sorted(str(t) for t in types))
                        table_multiple_ops.append((dir, ops_to_str, types_to_str, compile_time))

                size = os.path.getsize(filename)
                total_key_sizes += size
                if size > max_key_file_size:
                    big_key_files.append((dir, size, ops))

                nb_keys.setdefault(len(keydata.keys), 0)
                nb_keys[len(keydata.keys)] += 1
            except IOError:
                pass
            except AttributeError:
                    _logger.error(
                        "Could not read key file '%s'.",
                        filename)

    print_title("Theano cache: %s" % compiledir, overline='=', underline='=')
    print()

    print_title("List of %d compiled individual ops" % len(table), underline='+')
    print_title("sub dir/compiletime/Op/set of different associated Theano types", underline='-')
    table = sorted(table, key=lambda t: str(t[1]))
    for dir, op, types, compile_time in table:
        print(dir, '%.3fs' % compile_time, op, types)

    print()
    print_title("List of %d compiled sets of ops" % len(table_multiple_ops), underline='+')
    print_title("sub dir/compiletime/Set of ops/set of different associated Theano types", underline='-')
    table_multiple_ops = sorted(table_multiple_ops, key=lambda t: (t[1], t[2]))
    for dir, ops_to_str, types_to_str, compile_time in table_multiple_ops:
        print(dir, '%.3fs' % compile_time, ops_to_str, types_to_str)

    print()
    print_title(("List of %d compiled Op classes and "
                 "the number of times they got compiled" % len(table_op_class)), underline='+')
    table_op_class = sorted(iteritems(table_op_class), key=lambda t: t[1])
    for op_class, nb in table_op_class:
        print(op_class, nb)

    if big_key_files:
        big_key_files = sorted(big_key_files, key=lambda t: str(t[1]))
        big_total_size = sum([sz for _, sz, _ in big_key_files])
        print(("There are directories with key files bigger than %d bytes "
               "(they probably contain big tensor constants)" %
               max_key_file_size))
        print(("They use %d bytes out of %d (total size used by all key files)"
               "" % (big_total_size, total_key_sizes)))

        for dir, size, ops in big_key_files:
            print(dir, size, ops)

    nb_keys = sorted(iteritems(nb_keys))
    print()
    print_title("Number of keys for a compiled module", underline='+')
    print_title("number of keys/number of modules with that number of keys", underline='-')
    for n_k, n_m in nb_keys:
        print(n_k, n_m)
    print()
    print(("Skipped %d files that contained 0 op "
           "(are they always theano.scalar ops?)" % zeros_op))


def compiledir_purge():
    shutil.rmtree(config.compiledir)


def basecompiledir_ls():
    """
    Print list of files in the "theano.config.base_compiledir"
    """
    subdirs = []
    others = []
    for f in os.listdir(config.base_compiledir):
        if os.path.isdir(os.path.join(config.base_compiledir, f)):
            subdirs.append(f)
        else:
            others.append(f)

    subdirs = sorted(subdirs)
    others = sorted(others)

    print('Base compile dir is %s' % theano.config.base_compiledir)
    print('Sub-directories (possible compile caches):')
    for d in subdirs:
        print('    %s' % d)
    if not subdirs:
        print('    (None)')

    if others:
        print()
        print('Other files in base_compiledir:')
        for f in others:
            print('    %s' % f)


def basecompiledir_purge():
    shutil.rmtree(config.base_compiledir)
