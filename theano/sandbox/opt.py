"""
Optimizations addressing the ops in sandbox root directory
"""

import bisect
import logging

from theano.compile import optdb
from theano.gof import local_optimizer, EquilibriumDB
from theano.tensor.opt import register_specialize
from theano.sandbox.blocksparse import (
    SparseBlockGemv,
    SparseBlockOuter,
    sparse_block_gemv,
    sparse_block_outer,
    sparse_block_gemv_inplace,
    sparse_block_outer_inplace,
    CpuSparseBlockGemv,
    CpuSparseBlockOuter)


_logger = logging.getLogger('theano.sandbox.opt')


def _db_exists(db, db_name):
    """
        Tests whether the full path from `db_name[0]` down to
        `db_name[-1]` exists.

        Parameters
        ----------
        db: `theano.gof.optdb.DB`
            A dataset of optimisations or sub-datasets.
        db_name: list or tuple of strings
            Names of datasets from given one `db[db_name[0]]` down
            to the dataset of interest where to register.
            ex: ['level_1_dataset', 'level_2_dataset']
    """
    if len(db_name) == 1:
        return db_name[0] in db._names

    return db_name[0] in db._names and _db_exists(db[db_name[0]], db_name[1:])


def _db_register(db, db_name, *args):
    """
        Registers an object in last datasets given in db_name. `db_name[-1]`
        is deep in the hierarchy of `db`.

        Parameters
        ----------
        db: `theano.gof.optdb.DB`
            A dataset of optimisations or sub-datasets.
        db_name: list or tuple of strings
            Names of datasets from given one `db[db_name[0]]` down
            to the dataset of interest where to register.
            ex: ['level_1_dataset', 'level_2_dataset']
    """

    if len(db_name) == 0:
        return db.register(*args)

    return _db_register(db[db_name[0]], db_name[1:], *args)


def _db_positions(db, db_name, positions=()):
    """
        Returns the list of positions of all databases from `db_name[0]`
        down to `db_name[-1]`. The path is hierarchical, hence `db_name[0]`
        is in `db`, `db_name[1]` is in `db[db_name[0]]`, etc.

        Parameters
        ----------
        db: `theano.gof.optdb.DB`
            A dataset of optimisations or sub-datasets.
        db_name: list or tuple of strings
            Names of datasets from given one `db[db_name[0]]` down
            to the dataset of interests.
            ex: ['level_1_dataset', 'level_2_dataset']
    """
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
    op_class: `theano.gof.Op`
        A meta Op which have multiple implementations available
        for optimization.

    db_name: string, list or tuple of strings
        A string if optimization is inserted in `theano.compile.optdb`
        directly. List is used to insert an optimization deep inside a
        hierarchy of optimization databases.
    position: int or float
        Position of the optimisation in the target dataset.
        (Position in deep database if not optdb)
    *args
        Arguments to register the optimization.
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


@register_meta_opt(SparseBlockGemv, ["meta_cpu"], 51.0,
                   "fast_run", "fast_compile")
@local_optimizer([SparseBlockGemv])
def cpu_sparse_block_gemv_opt(node):
    """
        SparseBlockGemv -> CpuSparseBlockGemv
    """
    return [CpuSparseBlockGemv(node.op.inplace)(*node.inputs)]


@register_meta_opt(SparseBlockOuter, ["meta_cpu"], 51.0,
                   "fast_run", "fast_compile")
@local_optimizer([SparseBlockOuter])
def cpu_sparse_block_outer_opt(node):
    """
        SparseBlockOuter -> CpuSparseBlockOuter
    """
    return [CpuSparseBlockOuter(node.op.inplace)(*node.inputs)]


@register_specialize
@local_optimizer([sparse_block_gemv], inplace=True)
def local_inplace_block_sparse_gemv(node):
    """
        SparseBlockGemv(inplace=False) -> SparseBlockGemv(inplace=True)
    """
    return [sparse_block_gemv_inplace(*node.inputs)]


@register_specialize
@local_optimizer([sparse_block_outer], inplace=True)
def local_inplace_block_sparse_outer(node):
    """
        SparseBlockOuter(inplace=False) -> SparseBlockOuter(inplace=True)
    """
    return [sparse_block_outer_inplace(*node.inputs)]
