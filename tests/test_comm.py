# Copyright (c) Mike Jarvis and other collaborators
# See https://github.com/rmjarvis/MockMPI/LICENSE for license information.

from mock_mpi import mock_mpiexec
import numpy as np

def run_mpi_session(comm):
    """A simple MPI session we want to run in mock MPI mode.

    This serves as a test of comm.py and exec.py
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(rank, "rank, size = ", rank, size, flush=True)

    comm.send("Hello! My rank is %d." % rank, dest=0)
    print(rank, "sent hello message ", flush=True)

    if rank == 0:
        for p in range(size):
            print(rank, "try to read from ", p, flush=True)
            msg = comm.recv(source=p)
            print(rank, "received message: ", repr(msg), flush=True)

    print(rank, "Before barrier", flush=True)
    comm.Barrier()
    print(rank, "After barrier", flush=True)

    if rank == 0:
        data = np.arange(size) + 10
    else:
        data = None

    print(rank, "Before bcast: data = ", data, flush=True)
    data = comm.bcast(data, root=0)
    print(rank, "After bcast: data = ", data, flush=True)
    np.testing.assert_array_equal(data, np.arange(size) + 10)
    comm.Barrier()

    if rank == 0:
        data = np.arange(size) + 10
    else:
        data = np.empty(size, dtype=int)

    print(rank, "Before Bcast: data = ", data, flush=True)
    comm.Bcast(data, root=0)
    print(rank, "After Bcast: data = ", data, flush=True)
    np.testing.assert_array_equal(data, np.arange(size) + 10)
    comm.Barrier()

    if rank != 0:
        data = None

    print(rank, "Before scatter: data = ", data, flush=True)
    data = comm.scatter(data, root=0)
    print(rank, "After scatter: data = ", data, flush=True)
    assert data == rank + 10
    comm.Barrier()

    print(rank, "Before gather: data = ", data, flush=True)
    data = comm.gather(data, root=0)
    print(rank, "After gather: data = ", data, flush=True)
    if rank == 0:
        np.testing.assert_array_equal(data, np.arange(size) + 10)
    else:
        assert data is None
    comm.Barrier()

    data = np.arange(size) + rank ** 2 + 5
    print(rank, "Before alltoall: data = ", data, flush=True)
    data = comm.alltoall(data)
    print(rank, "After alltoall: data = ", data, flush=True)
    np.testing.assert_array_equal(data, np.arange(size) ** 2 + rank + 5)

    # test reduction
    if size < 52:
        letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        x = letters[rank]
        s = comm.reduce(x)
        if rank == 0:
            assert s == letters[:size]
        else:
            assert s is None
        comm.Barrier()
    else:
        print("Skipping reduction test - too large")

    # test all reduction
    if size < 52:
        letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        x = letters[rank]
        s = comm.allreduce(x)
        assert s == letters[:size]
        comm.Barrier()
    else:
        print("Skipping reduction test - too large")

    # test Reduce
    x = np.zeros(10, dtype=float) + rank
    y = np.zeros(10, dtype=float)
    print(rank, "Before Reduce: x = ", x, flush=True)
    print(rank, "Before Reduce: y = ", y, flush=True)
    comm.Reduce(x, y)
    if rank == 0:
        assert np.allclose(y, size * (size - 1) // 2)
    print(rank, "After Reduce: x = ", x, flush=True)
    print(rank, "After Reduce: y = ", y, flush=True)

    # # test All Reduce
    x = np.zeros(10, dtype=float) + rank
    y = np.zeros(10, dtype=float)
    print(rank, "Before AllReduce: x = ", x, flush=True)
    print(rank, "Before AllReduce: y = ", y, flush=True)
    comm.Allreduce(x, y)
    print(rank, "After AllReduce: x = ", x, flush=True)
    print(rank, "After AllReduce: y = ", y, flush=True)
    assert np.allclose(y, size * (size - 1) // 2)


def test_mpi_session():
    mock_mpiexec(2, run_mpi_session)
    mock_mpiexec(4, run_mpi_session)


if __name__ == "__main__":
    # Test this code.
    test_mpi_session()
