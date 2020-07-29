from mock_mpi import mock_mpiexec


def f(comm, txt, index=0, index2=0):
    assert txt == "abc"
    assert index == index2


def test_args():
    mock_mpiexec(2, f, args=("abc",))
    mock_mpiexec(2, f, args=["abc"])
    mock_mpiexec(
        2, f, args=["abc"], kwargs={"index": "cat", "index2": "cat"}
    )
