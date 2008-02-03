

def perform_linker(env, target = None):
    order = env.toposort()
    thunks = [op.perform for op in order]
    def ret():
        for thunk in thunks:
            thunk()
    if not target:
        return ret
    else:
        raise NotImplementedError("Cannot write thunk representation to a file.")


def perform_linker_nochecks(env, target = None):
    order = env.toposort()
    thunks = [op._perform for op in order]
    def ret():
        for thunk in thunks:
            thunk()
    if not target:
        return ret
    else:
        raise NotImplementedError("Cannot write thunk representation to a file.")


def cthunk_linker(env):
    order = env.toposort()
    thunks = []
    cstreak = []

    def append_cstreak():
        if cstreak:
            thunks.append(cutils.create_cthunk_loop(*cstreak))
            cstreak = []
    def ret():
        for thunk in thunks:
            thunk()

    for op in order:
        if hasattr(op, 'cthunk'):
            cstreak.append(op.cthunk())
        else:
            append_cstreak()
            thunks.append(op.perform)

    if len(thunks) == 1:
        return thunks[0]
    else:
        return ret

