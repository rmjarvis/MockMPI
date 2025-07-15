# Copyright (c) Mike Jarvis and other collaborators
# See https://github.com/rmjarvis/MockMPI/LICENSE for license information.

import numpy as np

# This constant used to be 1 in both MPICH and OpenMPI,
# but starting with mpi4py version 4, they switched it to -1.
# Even worse, starting with 4.1, it's not a subtype of int as it had been until then.
# But it does cast to an int if requested.
# Use -1 here, but when we check for it allow 1 as well.
# And if we happen to have mpi4py installed, include whatever it actually has as well
# both for the value and the type.
try:
    from mpi4py.MPI import IN_PLACE
    ALLOWED_IN_PLACE_TYPES = (int, type(IN_PLACE))
except ImportError:
    IN_PLACE = -1
    ALLOWED_IN_PLACE_TYPES = (int,)
ALLOWED_IN_PLACE = [IN_PLACE, 1, -1]


class MockComm(object):
    """A class to mock up the MPI Comm API using a multiprocessing Pipe.

    """

    def __init__(self, rank, size, pipes, barrier):
        self.rank = rank
        self.size = size
        self.pipes = pipes
        self.barrier = barrier

    def __bool__(self):
        return self.size > 0

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return self.size

    def send(self, msg, dest):
        if dest != self.rank:
            self.pipes[dest].send(msg)
        else:
            self.msg = msg

    def Send(self, msg, dest):
        if not isinstance(msg, np.ndarray):
            raise ValueError(
                "Can only use Send with numpy arrays "
                "(Mocking code does not handle general buffers)"
            )
        self.send(msg, dest)

    def recv(self, source):
        if source != self.rank:
            msg = self.pipes[source].recv()
        else:
            msg = self.msg
        return msg

    def Recv(self, buffer, source):
        msg = self.recv(source)
        buffer[:] = msg

    def Barrier(self):
        self.barrier.wait()

    def bcast(self, msg, root=0):
        if root == self.rank:
            for p in range(self.size):
                self.send(msg, p)
        msg = self.recv(root)
        return msg

    def Bcast(self, msg, root=0):
        if root == self.rank:
            for p in range(self.size):
                self.Send(msg, p)
        self.Recv(msg, root)

    def scatter(self, data, root=0):
        if root == self.rank:
            for p in range(self.size):
                self.send(data[p], p)
        data = self.recv(root)
        return data

    def gather(self, data, root=0):
        self.send(data, root)
        if root == self.rank:
            new_data = []
            for p in range(self.size):
                new_data.append(self.recv(p))
            return new_data
        else:
            return None

    def alltoall(self, data=0):
        for p in range(self.size):
            self.send(data[p], p)
        new_data = []
        for p in range(self.size):
            new_data.append(self.recv(p))
        return new_data

    def reduce(self, sendobj, op=None, root=0):
        if op is not None:
            raise NotImplementedError("Not implemented non-sum reductions in mock MPI")
        new_data = self.gather(sendobj, root)

        if root == self.rank:
            d = new_data[0]
            for d2 in new_data[1:]:
                d = d + d2
            return d
        else:
            return None

    def allreduce(self, sendobj, op=None):
        d = self.reduce(sendobj, op)
        d = self.bcast(d)
        return d

    def Reduce(self, sendbuf, recvbuf, op=None, root=0):
        if isinstance(sendbuf, ALLOWED_IN_PLACE_TYPES) and (sendbuf in ALLOWED_IN_PLACE):
            sendbuf = recvbuf.copy()

        if not isinstance(sendbuf, np.ndarray):
            raise ValueError(
                "Cannot use Reduce with non-arrays. "
                "(Mocking code does not handle general buffers)"
            )

        r = self.reduce(sendbuf, op=op, root=root)
        if self.rank == root:
            recvbuf[:] = r

    def Allreduce(self, sendbuf, recvbuf, op=None):
        self.Reduce(sendbuf, recvbuf, op)
        self.Bcast(recvbuf)

    def allgather(self, sendobj):
        obj = self.gather(sendobj)
        return self.bcast(obj)

    # Instance methods not implemented
    def Abort(self, *args, **kwargs):
        raise NotImplementedError("The method 'Abort' is not implemented in mockmpi")

    def Accept(self, *args, **kwargs):
        raise NotImplementedError("The method 'Accept' is not implemented in mockmpi")

    def Allgather(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Allgather' is not implemented in mockmpi"
        )

    def Allgatherv(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Allgatherv' is not implemented in mockmpi"
        )

    def Alltoall(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Alltoall' is not implemented in mockmpi"
        )

    def Alltoallv(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Alltoallv' is not implemented in mockmpi"
        )

    def Alltoallw(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Alltoallw' is not implemented in mockmpi"
        )

    def Bsend(self, *args, **kwargs):
        raise NotImplementedError("The method 'Bsend' is not implemented in mockmpi")

    def Bsend_init(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Bsend_init' is not implemented in mockmpi"
        )

    def Call_errhandler(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Call_errhandler' is not implemented in mockmpi"
        )

    def Cart_map(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Cart_map' is not implemented in mockmpi"
        )

    def Clone(self, *args, **kwargs):
        raise NotImplementedError("The method 'Clone' is not implemented in mockmpi")

    def Connect(self, *args, **kwargs):
        raise NotImplementedError("The method 'Connect' is not implemented in mockmpi")

    def Create(self, *args, **kwargs):
        raise NotImplementedError("The method 'Create' is not implemented in mockmpi")

    def Create_cart(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Create_cart' is not implemented in mockmpi"
        )

    def Create_dist_graph(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Create_dist_graph' is not implemented in mockmpi"
        )

    def Create_dist_graph_adjacent(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Create_dist_graph_adjacent' is not implemented in mockmpi"
        )

    def Create_graph(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Create_graph' is not implemented in mockmpi"
        )

    def Create_group(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Create_group' is not implemented in mockmpi"
        )

    def Create_intercomm(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Create_intercomm' is not implemented in mockmpi"
        )

    def Delete_attr(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Delete_attr' is not implemented in mockmpi"
        )

    def Disconnect(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Disconnect' is not implemented in mockmpi"
        )

    def Dup(self, *args, **kwargs):
        raise NotImplementedError("The method 'Dup' is not implemented in mockmpi")

    def Dup_with_info(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Dup_with_info' is not implemented in mockmpi"
        )

    def Exscan(self, *args, **kwargs):
        raise NotImplementedError("The method 'Exscan' is not implemented in mockmpi")

    def Free(self, *args, **kwargs):
        raise NotImplementedError("The method 'Free' is not implemented in mockmpi")

    def Gather(self, *args, **kwargs):
        raise NotImplementedError("The method 'Gather' is not implemented in mockmpi")

    def Gatherv(self, *args, **kwargs):
        raise NotImplementedError("The method 'Gatherv' is not implemented in mockmpi")

    def Get_attr(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Get_attr' is not implemented in mockmpi"
        )

    def Get_errhandler(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Get_errhandler' is not implemented in mockmpi"
        )

    def Get_group(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Get_group' is not implemented in mockmpi"
        )

    def Get_info(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Get_info' is not implemented in mockmpi"
        )

    def Get_name(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Get_name' is not implemented in mockmpi"
        )

    def Get_topology(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Get_topology' is not implemented in mockmpi"
        )

    def Graph_map(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Graph_map' is not implemented in mockmpi"
        )

    def Iallgather(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Iallgather' is not implemented in mockmpi"
        )

    def Iallgatherv(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Iallgatherv' is not implemented in mockmpi"
        )

    def Iallreduce(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Iallreduce' is not implemented in mockmpi"
        )

    def Ialltoall(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Ialltoall' is not implemented in mockmpi"
        )

    def Ialltoallv(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Ialltoallv' is not implemented in mockmpi"
        )

    def Ialltoallw(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Ialltoallw' is not implemented in mockmpi"
        )

    def Ibarrier(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Ibarrier' is not implemented in mockmpi"
        )

    def Ibcast(self, *args, **kwargs):
        raise NotImplementedError("The method 'Ibcast' is not implemented in mockmpi")

    def Ibsend(self, *args, **kwargs):
        raise NotImplementedError("The method 'Ibsend' is not implemented in mockmpi")

    def Idup(self, *args, **kwargs):
        raise NotImplementedError("The method 'Idup' is not implemented in mockmpi")

    def Iexscan(self, *args, **kwargs):
        raise NotImplementedError("The method 'Iexscan' is not implemented in mockmpi")

    def Igather(self, *args, **kwargs):
        raise NotImplementedError("The method 'Igather' is not implemented in mockmpi")

    def Igatherv(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Igatherv' is not implemented in mockmpi"
        )

    def Improbe(self, *args, **kwargs):
        raise NotImplementedError("The method 'Improbe' is not implemented in mockmpi")

    def Iprobe(self, *args, **kwargs):
        raise NotImplementedError("The method 'Iprobe' is not implemented in mockmpi")

    def Irecv(self, *args, **kwargs):
        raise NotImplementedError("The method 'Irecv' is not implemented in mockmpi")

    def Ireduce(self, *args, **kwargs):
        raise NotImplementedError("The method 'Ireduce' is not implemented in mockmpi")

    def Ireduce_scatter(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Ireduce_scatter' is not implemented in mockmpi"
        )

    def Ireduce_scatter_block(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Ireduce_scatter_block' is not implemented in mockmpi"
        )

    def Irsend(self, *args, **kwargs):
        raise NotImplementedError("The method 'Irsend' is not implemented in mockmpi")

    def Is_inter(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Is_inter' is not implemented in mockmpi"
        )

    def Is_intra(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Is_intra' is not implemented in mockmpi"
        )

    def Iscan(self, *args, **kwargs):
        raise NotImplementedError("The method 'Iscan' is not implemented in mockmpi")

    def Iscatter(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Iscatter' is not implemented in mockmpi"
        )

    def Iscatterv(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Iscatterv' is not implemented in mockmpi"
        )

    def Isend(self, *args, **kwargs):
        raise NotImplementedError("The method 'Isend' is not implemented in mockmpi")

    def Issend(self, *args, **kwargs):
        raise NotImplementedError("The method 'Issend' is not implemented in mockmpi")

    def Mprobe(self, *args, **kwargs):
        raise NotImplementedError("The method 'Mprobe' is not implemented in mockmpi")

    def Probe(self, *args, **kwargs):
        raise NotImplementedError("The method 'Probe' is not implemented in mockmpi")

    def Recv_init(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Recv_init' is not implemented in mockmpi"
        )

    def Reduce_scatter(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Reduce_scatter' is not implemented in mockmpi"
        )

    def Reduce_scatter_block(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Reduce_scatter_block' is not implemented in mockmpi"
        )

    def Rsend(self, *args, **kwargs):
        raise NotImplementedError("The method 'Rsend' is not implemented in mockmpi")

    def Rsend_init(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Rsend_init' is not implemented in mockmpi"
        )

    def Scan(self, *args, **kwargs):
        raise NotImplementedError("The method 'Scan' is not implemented in mockmpi")

    def Scatter(self, *args, **kwargs):
        raise NotImplementedError("The method 'Scatter' is not implemented in mockmpi")

    def Scatterv(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Scatterv' is not implemented in mockmpi"
        )

    def Send_init(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Send_init' is not implemented in mockmpi"
        )

    def Sendrecv(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Sendrecv' is not implemented in mockmpi"
        )

    def Sendrecv_replace(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Sendrecv_replace' is not implemented in mockmpi"
        )

    def Set_attr(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Set_attr' is not implemented in mockmpi"
        )

    def Set_errhandler(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Set_errhandler' is not implemented in mockmpi"
        )

    def Set_info(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Set_info' is not implemented in mockmpi"
        )

    def Set_name(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Set_name' is not implemented in mockmpi"
        )

    def Spawn(self, *args, **kwargs):
        raise NotImplementedError("The method 'Spawn' is not implemented in mockmpi")

    def Spawn_multiple(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Spawn_multiple' is not implemented in mockmpi"
        )

    def Split(self, *args, **kwargs):
        raise NotImplementedError("The method 'Split' is not implemented in mockmpi")

    def Split_type(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Split_type' is not implemented in mockmpi"
        )

    def Ssend(self, *args, **kwargs):
        raise NotImplementedError("The method 'Ssend' is not implemented in mockmpi")

    def Ssend_init(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Ssend_init' is not implemented in mockmpi"
        )

    def barrier(self, *args, **kwargs):
        raise NotImplementedError("The method 'barrier' is not implemented in mockmpi")

    def bsend(self, *args, **kwargs):
        raise NotImplementedError("The method 'bsend' is not implemented in mockmpi")

    def exscan(self, *args, **kwargs):
        raise NotImplementedError("The method 'exscan' is not implemented in mockmpi")

    def ibsend(self, *args, **kwargs):
        raise NotImplementedError("The method 'ibsend' is not implemented in mockmpi")

    def improbe(self, *args, **kwargs):
        raise NotImplementedError("The method 'improbe' is not implemented in mockmpi")

    def iprobe(self, *args, **kwargs):
        raise NotImplementedError("The method 'iprobe' is not implemented in mockmpi")

    def irecv(self, *args, **kwargs):
        raise NotImplementedError("The method 'irecv' is not implemented in mockmpi")

    def isend(self, *args, **kwargs):
        raise NotImplementedError("The method 'isend' is not implemented in mockmpi")

    def issend(self, *args, **kwargs):
        raise NotImplementedError("The method 'issend' is not implemented in mockmpi")

    def mprobe(self, *args, **kwargs):
        raise NotImplementedError("The method 'mprobe' is not implemented in mockmpi")

    def probe(self, *args, **kwargs):
        raise NotImplementedError("The method 'probe' is not implemented in mockmpi")

    def py2f(self, *args, **kwargs):
        raise NotImplementedError("The method 'py2f' is not implemented in mockmpi")

    def scan(self, *args, **kwargs):
        raise NotImplementedError("The method 'scan' is not implemented in mockmpi")

    def sendrecv(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'sendrecv' is not implemented in mockmpi"
        )

    def ssend(self, *args, **kwargs):
        raise NotImplementedError("The method 'ssend' is not implemented in mockmpi")

    # Properties not implemented

    @property
    def topology(self):
        raise NotImplementedError(
            "The property 'topology' is not implemented in mockmpi"
        )

    @property
    def group(self):
        raise NotImplementedError("The property 'group' is not implemented in mockmpi")

    @property
    def name(self):
        raise NotImplementedError("The property 'name' is not implemented in mockmpi")

    @property
    def is_inter(self):
        raise NotImplementedError(
            "The property 'is_inter' is not implemented in mockmpi"
        )

    @property
    def is_intra(self):
        raise NotImplementedError(
            "The property 'is_intra' is not implemented in mockmpi"
        )

    @property
    def is_topo(self):
        raise NotImplementedError(
            "The property 'is_topo' is not implemented in mockmpi"
        )

    # 'Info' is the only writeable property
    @property
    def info(self):
        raise NotImplementedError("The property 'info' is not implemented in mockmpi")

    @info.setter
    def info(self, *args, **kwargs):
        raise NotImplementedError("The property 'info' is not implemented in mockmpi")

    # Class methods not yet implemented
    @classmethod
    def Compare(cls, *args, **kwargs):
        raise NotImplementedError(
            "The class method 'Compare' is not implemented in mockmpi"
        )

    @classmethod
    def Get_parent(cls, *args, **kwargs):
        raise NotImplementedError(
            "The class method 'Get_parent' is not implemented in mockmpi"
        )

    @classmethod
    def Join(cls, *args, **kwargs):
        raise NotImplementedError(
            "The class method 'Join' is not implemented in mockmpi"
        )

    @classmethod
    def Create_keyval(cls, *args, **kwargs):
        raise NotImplementedError(
            "The class method 'Create_keyval' is not implemented in mockmpi"
        )

    @classmethod
    def Free_keyval(cls, *args, **kwargs):
        raise NotImplementedError(
            "The class method 'Free_keyval' is not implemented in mockmpi"
        )

    @classmethod
    def f2py(cls, *args, **kwargs):
        raise NotImplementedError(
            "The class method 'f2py' is not implemented in mockmpi"
        )
