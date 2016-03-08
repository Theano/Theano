from __future__ import print_function
import six.moves.cPickle as pickle
import logging
import os
import shutil

import numpy

import theano
from six import string_types, iteritems
from theano.configparser import config
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
                            if isinstance(obj, numpy.ndarray):
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

                except EOFError:
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


def print_compiledir_content():
    max_key_file_size = 1 * 1024 * 1024  # 1M

    compiledir = theano.config.compiledir
    table = []
    more_than_one_ops = 0
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
                if len(ops) == 0:
                    zeros_op += 1
                elif len(ops) > 1:
                    more_than_one_ops += 1
                else:
                    types = list(set([x for x in flatten(keydata.keys)
                                      if isinstance(x, theano.gof.Type)]))
                    table.append((dir, ops[0], types))

                size = os.path.getsize(filename)
                total_key_sizes += size
                if size > max_key_file_size:
                    big_key_files.append((dir, size, ops))

                nb_keys.setdefault(len(keydata.keys), 0)
                nb_keys[len(keydata.keys)] += 1
            except IOError:
                pass

    print("List of %d compiled individual ops in this theano cache %s:" % (
        len(table), compiledir))
    print("sub directory/Op/a set of the different associated Theano type")
    table = sorted(table, key=lambda t: str(t[1]))
    table_op_class = {}
    for dir, op, types in table:
        print(dir, op, types)
        table_op_class.setdefault(op.__class__, 0)
        table_op_class[op.__class__] += 1

    print()
    print(("List of %d individual compiled Op classes and "
           "the number of times they got compiled" % len(table_op_class)))
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
    print("Number of keys for a compiled module")
    print("number of keys/number of modules with that number of keys")
    for n_k, n_m in nb_keys:
        print(n_k, n_m)

    print(("Skipped %d files that contained more than"
           " 1 op (was compiled with the C linker)" % more_than_one_ops))
    print(("Skipped %d files that contained 0 op "
           "(are they always theano.scalar ops?)" % zeros_op))


def compiledir_purge():
    shutil.rmtree(config.compiledir)


def basecompiledir_ls():
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
