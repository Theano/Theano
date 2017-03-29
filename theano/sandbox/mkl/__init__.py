from __future__ import absolute_import, print_function, division
import os
import errno
import sys
import logging
import textwrap
import stat
import shutil
import theano
from theano import config, gof
from theano.gof.cmodule import Compiler, get_lib_extension, GCC_compiler
from theano.sandbox.mkl.mkl_helper import header_text
from theano.gof.compilelock import get_lock, release_lock

from theano.gof import EquilibriumDB, SequenceDB

from theano.tensor.blas import ldflags

_logger_name = 'theano.sandbox.mkl'
_logger = logging.getLogger(_logger_name)


mkl_optimizer = EquilibriumDB(ignore_newtrees=False)
mkl_seqopt = SequenceDB()


def register_opt(*tags, **kwargs):
    if any([not isinstance(t, str) for t in tags]):
        raise RuntimeError("Bad call to register_opt."
                           " All tags must be strings.", tags)

    def f(local_opt):
        name = (kwargs and kwargs.pop('name')) or local_opt.__name__
        mkl_optimizer.register(name, local_opt, 'fast_run', 'fast_compile',
                               'mkl', *tags, **kwargs)
        return local_opt
    return f


class MKLVersion(gof.Op):
    def c_header_dirs(self):
        return ldflags(libs=False, include_dir=True)

    def c_libraries(self):
        return ldflags()

    def c_lib_dirs(self):
        return ldflags(libs=False, libs_dir=True)

    def make_node(self):
        return gof.Apply(self, [], [gof.Generic()()])

    def c_support_code(self):
        ccode = header_text()
        ccode += """
        #if PY_MAJOR_VERSION >= 3
            #define PyInt_FromLong PyLong_FromLong
        #endif
        """
        return ccode

    def c_code(self, node, name, inputs, outputs, sub):
        o = outputs[0]
        return textwrap.dedent(
            """
            MKLVersion v;
            mkl_get_version(&v);
            %(o)s = PyInt_FromLong(atoi(v.Build));
            """) % locals()

    def c_code_cache_version(self):
        # Not needed, but make it clear that we do not want to cache this.
        return None


def mkl_version():
    """
    Return the current mkl version (e.g., 20160701) we compile with.
    While `0` indicate fail to get mkl version.
    """
    if mkl_version.v is None:
        try:
            f = theano.function([], MKLVersion()(),
                                theano.Mode(optimizer=None),
                                profile=False)
            mkl_version.v = f()
        except:
            mkl_version.v = 0
    return mkl_version.v

mkl_version.v = None


def _dnn_is_able_to_link():
    if _dnn_is_able_to_link.status is None:
        preambule = """
            #include <stdio.h>
            #include <mkl_dnn.h>
            """
        preambule += textwrap.dedent(header_text())

        body = textwrap.dedent(
            """
            dnnError_t err;
            dnnLayout_t usr_layout = NULL;
            size_t size[1] = {256};
            size_t stride[1] = {1};

            if ((err = dnnLayoutCreate_F32(&usr_layout, 1, size, stride)) != E_SUCCESS) {
                fprintf(stderr, "Failed to create user layout with mkl: %s", err);
                return (-1);
            }
            """)

        if 'mklml_intel' in config.blas.ldflags:
            params = ['-l', 'mklml_intel', '-l', 'iomp5']
        elif 'mklml_gnu' in config.blas.ldflags:
            params = ['-l', 'mklml_gnu', '-l', 'iomp5']
        elif 'mkl_rt' in config.blas.ldflags:
            params = ['-l', 'mkl_rt']
        else:
            params = []

        compilation_ok, run_ok, out, err = Compiler._try_flags(
            flag_list=params, preambule=preambule, body=body,
            try_run=True, output=True, compiler=theano.config.cxx,
            comp_args=False)

        mkl_available.msg = (
            "Can not compile with MKL. We got this error: " +
            str(err))
        _dnn_is_able_to_link.status = run_ok

    return _dnn_is_able_to_link.status

_dnn_is_able_to_link.status = None


def _dnn_is_enabled_by_user():
    if config.device != "cpu":
        mkl_available.msg = "MKL is disabled since device is not CPU. " \
                            "Set the device to 'CPU' if you want to use MKL."
        return False

    if config.mkl.lib != "mkl":
        raise NotImplementedError("MKL lib only supports 'mkl' currently, "
                                  "but got %s." % config.mkl.lib)

    if config.mkl.nn.enabled == "False":
        mkl_available.msg = "MKL is disabled by the 'mkl.nn.enabled' setting."
        return False

    return True


def mkl_available():
    # Check this status every time which allows user
    # to change their setting in different stage of model
    if _dnn_is_enabled_by_user() is not True:
        return False

    if _dnn_is_able_to_link() is not True:
        if config.mkl.nn.enabled == "True":
            raise RuntimeError("You enabled MKL, but we aren't able "
                               "to use it, %s" % mkl_available.msg)
        return False

    return True

mkl_available.msg = None


if mkl_available():
    mkl_path = os.path.abspath(os.path.split(__file__)[0])
    mkl_ndarray_loc = os.path.join(config.compiledir, 'mkl_ndarray')
    mkl_ndarray_so = os.path.join(
        mkl_ndarray_loc, 'mkl_ndarray.' + get_lib_extension())
    libmkl_ndarray_so = os.path.join(
        mkl_ndarray_loc, 'libmkl_ndarray.' + get_lib_extension())

    def try_import_mkl_ndarray():
        mkl_files = ('mkl_ndarray.c', 'mkl_ndarray.h')

        stat_times = [os.stat(os.path.join(mkl_path, mkl_file))[stat.ST_MTIME]
                      for mkl_file in mkl_files]
        date = max(stat_times)
        if os.path.exists(mkl_ndarray_so):
            if date >= os.stat(mkl_ndarray_so)[stat.ST_MTIME]:
                return False

        try:
            sys.path[0:0] = [config.compiledir]
            import mkl_ndarray.mkl_ndarray  # noqa
            del sys.path[0]
        except ImportError:
            return False
        return True

    compile_mkl_ndarray = not try_import_mkl_ndarray()

    if compile_mkl_ndarray:
        get_lock()
        try:
            if not try_import_mkl_ndarray():
                try:
                    code = open(os.path.join(mkl_path, 'mkl_ndarray.c')).read()
                    if not os.path.exists(mkl_ndarray_loc):
                        os.makedirs(mkl_ndarray_loc)
                    compiler = GCC_compiler()
                    preargs = compiler.compile_args()

                    if 'mklml_intel' in config.blas.ldflags:
                        mkl_lib = 'mklml_intel'
                    elif 'mklml_gnu' in config.blas.ldflags:
                        mkl_lib = 'mklml_gnu'
                    else:
                        mkl_lib = 'mkl_rt'
                    compiler.compile_str('mkl_ndarray',
                                         code,
                                         location=mkl_ndarray_loc,
                                         include_dirs=[mkl_path],
                                         libs=[mkl_lib],
                                         preargs=preargs,)

                    from mkl_ndarray.mkl_ndarray import *  # noqa
                except Exception as e:
                    _logger.error('Failed to compile mkl_ndarray.c: %s', str(e))
        finally:
            release_lock()

    def ok():
        try:
            open(libmkl_ndarray_so).close()
            return True
        except IOError:
            return False

    if not ok():
        if sys.platform == "win32":
            shutil.copyfile(mkl_ndarray_so, libmkl_ndarray_so)
        else:
            try:
                os.symlink(mkl_ndarray_so, libmkl_ndarray_so)
            except OSError as e:
                if getattr(e, 'errno', None) != errno.EEXIST or not ok():
                    raise


# register name of 'mkl_opt' in opt.py and then add tags for it.
try:
    from . import opt
    opt.optdb.add_tags('mkl_opt', 'fast_compile', 'fast_run')
except:
    pass
