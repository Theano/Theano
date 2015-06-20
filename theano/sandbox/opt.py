"""
Optimizations addressing the ops in sandbox root directory
"""

import bisect
import logging

from theano.compile import optdb
from theano.gof import local_optimizer, EquilibriumDB
from theano.sandbox.blocksparse import (
    SparseBlockGemv, 
    SparseBlockOuter,
    cpu_sparse_block_gemv,
    cpu_sparse_block_outer)


_logger = logging.getLogger('theano.sandbox.opt')


def _db_exists(db, db_name):
    if len(db_name) == 1:
        return db_name[0] in db._names
    
    return db_name[0] in db._names and _db_exists(db[db_name[0]], db_name[1:])


def _db_register(db, db_name, *args):
    if len(db_name) == 0:
        return db.register(*args)

    return _db_register(db[db_name[0]], db_name[1:], *args)


def _db_positions(db, db_name, positions=()):
    if len(db_name) == 0:
        return positions

    db_position = db.__position__.get(db_name[0], 0.)

    return _db_positions(db[db_name[0]], db_name[1:], 
                        positions + (db_position, ))


def register_meta_opt(op_class, db_name, position, *args):
    """
    Registers a given optimization under given database name and saves 
    optimization information in `op_class.registered_opts`.

    Parameters
    ----------
    op_class: `Op` ?

    db_name: string, list or tuple

    position: int or float
        ?
    args: ?

    Returns
    -------
    ?
    """

    if isinstance(db_name, str):
        db_name = [db_name]

    def call(local_meta_opt):
        if not _db_exists(optdb, db_name):
            # TODO: Would another default DB be better?
            _db_register(optdb, db_name[:-2], 
                        db_name[-1], EquilibriumDB(), position, *args)

        _db_register(optdb, db_name,
                    local_meta_opt.__name__, local_meta_opt, *args)

        positions = _db_positions(optdb, db_name)

        idx = bisect.bisect_left((positions, local_meta_opt),
                                 op_class.registered_opts)
        op_class.registered_opts.insert(idx, 
                (positions, local_meta_opt.__name__))

        return local_meta_opt

    return call


@register_meta_opt(SparseBlockGemv, ["meta_cpu"], 51.0, "fast_run", "fast_compile")
@local_optimizer([SparseBlockGemv])
def cpu_sparse_block_gemv_opt(node):
    """
        TODO: WRITEME
    """
    if node.op.inplace:
        _logger.warning("CPU version of sparse_block_gemv does not support"
                        "inplace")

    return [cpu_sparse_block_gemv(*node.inputs)]


@register_meta_opt(SparseBlockOuter, ["meta_cpu"], 51.0, "fast_run", "fast_compile")
@local_optimizer([SparseBlockOuter])
def cpu_sparse_block_outer_opt(node):
    """
        TODO: WRITEME
    """
    if node.op.inplace:
        _logger.warning("CPU version of sparse_block_outer does not support"
                        "inplace")

    return [cpu_sparse_block_outer(*node.inputs)]
