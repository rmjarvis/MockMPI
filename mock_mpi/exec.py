# Copyright (c) Mike Jarvis and other collaborators
# See https://github.com/rmjarvis/MockMPI/LICENSE for license information.

import numpy as np
import multiprocessing as mp
from .comm import MockComm

# We use this subclass to help with exception
# handling, as described here:
# https://stackoverflow.com/a/33599967/989692
class Process(mp.Process):
    def __init__(self, *args, **kwargs):
        mp.Process.__init__(self, *args, **kwargs)
        self._pconn, self._cconn = mp.Pipe()
        self._exception = None

    def run(self):
        try:
            mp.Process.run(self)
            self._cconn.send(None)
        except Exception as e:
            self._cconn.send(e)
            raise e

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception


def mock_mpiexec(nproc, target, *args, **kwargs):
    """Run a function, given as target, as though it were an MPI session using mpiexec -n nproc
    but using multiprocessing instead of mpi.
    """
    mp.set_start_method("spawn", force=True)

    # Make the message passing pipes
    all_pipes = [{} for p in range(nproc)]
    for i in range(nproc):
        for j in range(i + 1, nproc):
            p1, p2 = mp.Pipe()
            all_pipes[i][j] = p1
            all_pipes[j][i] = p2

    # Make a barrier
    barrier = mp.Barrier(nproc)

    # Make fake MPI-like comm object
    comms = [
        MockComm(rank, nproc, pipes, barrier) for rank, pipes in enumerate(all_pipes)
    ]

    # Make processes
    procs = [Process(target=target, args=(comm,) + args, kwargs=kwargs) for comm in comms]

    for p in procs:
        p.start()

    for p in procs:
        d = p.join()
        if p.exception:
            raise p.exception.__class__ from p.exception
