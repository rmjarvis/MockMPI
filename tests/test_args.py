from mockmpi import mock_mpiexec


def f(comm, txt, index=0, index2=0):
    assert txt == "abc"
    assert index == index2


def test_args():
    # direct args
    mock_mpiexec(2, f, "abc")
    mock_mpiexec(2, f, "abc", index="cat", index2="cat")

    # if have args as list and kwargs as dict, can dereference them.
    args = ["abc"]
    mock_mpiexec(2, f, *args)
    kwargs = {"index": "cat", "index2": "cat"}
    mock_mpiexec(2, f, *args, **kwargs)

    # can also dereference as literals, which is parobably weird, but allowed.
    mock_mpiexec(2, f, *["abc"], **{"index": "cat", "index2": "cat"})

if __name__ == '__main__':
    test_args()
