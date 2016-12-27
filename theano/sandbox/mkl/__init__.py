from __future__ import absolute_import, print_function, division
import logging
import textwrap

import theano
from theano import config, gof
from six import integer_types
from theano.tensor.blas import ldflags
from theano.gof.cmodule import Compiler
from theano.sandbox.mkl import mkl_helper

from theano.gof import EquilibriumDB, SequenceDB

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
    def c_headers(self):
        return super(MKLVersion, self).c_headers()

    def c_libraries(self):
        return ldflags()

    def make_node(self):
        return gof.Apply(self, [], [gof.Generic()()])

    def c_support_code(self):
        return mkl_helper.header_text()

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
    """
    if not mkl_available():
        raise Exception(
            "We can't determine the mkl version as it is not available",
            mkl_available.msg)

    if mkl_version.v is None:
        f = theano.function([], MKLVersion()(),
                            theano.Mode(optimizer=None),
                            profile=False)
        mkl_version.v = f()
    return mkl_version.v


mkl_version.v = None


def mkl_available():
    if config.device != "cpu":
        mkl_available.avail = False
        mkl_available.msg = "MKL is disabled since device is not CPU. " \
                            "Set the device to 'CPU' if you want to use MKL."
        return mkl_available.avail

    if config.dnn.enabled == "False":
        mkl_available.avail = False
        mkl_available.msg = "MKL is disabled by the 'dnn.enabled' setting."
        return mkl_available.avail

    if config.dnn.enabled == "cudnn":
        if config.device == "cpu":
            mkl_available.avail = None
            config.dnn.enabled = "auto"
            print('WARNING: when device is cpu, config.dnn.enabled=cudnn is not supported, '
                  'Swithch to "auto" flag.')
            # FIXME call python warning module
        else:
            mkl_available.avail = False
            mkl_available.msg = "Disabled by dnn.enabled flag"
            return mkl_available.avail

    if (config.dnn.enabled == "auto" and config.device == "cpu") or config.dnn.enabled == "mkl":
        if mkl_available.avail is None:
            preambule = """
                #include <stdio.h>
             """
            preambule += textwrap.dedent(mkl_helper.header_text())

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
            params = [theano.config.blas.ldflags]

            comp, out, err = Compiler._try_flags(
                flag_list=params, preambule=preambule, body=body,
                try_run=False, output=True, compiler=theano.config.cxx, comp_args=False)

            mkl_available.avail = comp
            if mkl_available.avail is False:
                mkl_available.msg = (
                    "Can not compile with MKL. We got this error:\n" +
                    str(err))
            else:
                # If we can compile, check that we can import and run.
                v = mkl_version()
                if not isinstance(v, integer_types):
                    mkl_available.avail = False
                    mkl_available.msg = ("Got incorrect mkl version format")
                    raise RuntimeError(mkl_available.msg)
                if v == -1 or v < 20160701:  # FIXME, check the version for first mkl primitive
                    mkl_available.avail = False
                    mkl_available.msg = "Version(%d) is too old, please update the newer one after version %d." % (v, int(20160701))  # FIXME, check the version for the first mkl primitive
                    raise RuntimeError(mkl_available.msg)
                else:
                    mkl_available.avail = comp
        else:
            return mkl_available.avail

    '''
    ## leave mkl-dnn here for future use
    if config.dnn.enabled == "mkl-dnn":
        if not mkl_available.avail:
            raise NotImplemented(
                "mkl-dnn is not supported, %s" % mkl_available.msg)
    '''
    return mkl_available.avail


mkl_available.avail = None
mkl_available.msg = None


# register name of 'mkl_opt' in opt.py and then add tags for it.
if mkl_available():
    from . import opt
    opt.optdb.add_tags('mkl_opt', 'fast_compile', 'fast_run')
