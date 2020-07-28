# Copyright (c) 2003-2020 by Mike Jarvis
#
# mock_mpi is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions, and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.

import numpy as np

# This constant seems to have the same value in MPICH and OpenMPI
# so we reproduce it here since it can be quite important.
IN_PLACE = 1


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
        if sendbuf is IN_PLACE:
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

    # Instance methods not implemented
    def Abort(self, *args, **kwargs):
        raise NotImplementedError("The method 'Abort' is not implemented in mock_mpi")

    def Accept(self, *args, **kwargs):
        raise NotImplementedError("The method 'Accept' is not implemented in mock_mpi")

    def Allgather(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Allgather' is not implemented in mock_mpi"
        )

    def Allgatherv(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Allgatherv' is not implemented in mock_mpi"
        )

    def Alltoall(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Alltoall' is not implemented in mock_mpi"
        )

    def Alltoallv(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Alltoallv' is not implemented in mock_mpi"
        )

    def Alltoallw(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Alltoallw' is not implemented in mock_mpi"
        )

    def Bsend(self, *args, **kwargs):
        raise NotImplementedError("The method 'Bsend' is not implemented in mock_mpi")

    def Bsend_init(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Bsend_init' is not implemented in mock_mpi"
        )

    def Call_errhandler(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Call_errhandler' is not implemented in mock_mpi"
        )

    def Cart_map(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Cart_map' is not implemented in mock_mpi"
        )

    def Clone(self, *args, **kwargs):
        raise NotImplementedError("The method 'Clone' is not implemented in mock_mpi")

    def Connect(self, *args, **kwargs):
        raise NotImplementedError("The method 'Connect' is not implemented in mock_mpi")

    def Create(self, *args, **kwargs):
        raise NotImplementedError("The method 'Create' is not implemented in mock_mpi")

    def Create_cart(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Create_cart' is not implemented in mock_mpi"
        )

    def Create_dist_graph(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Create_dist_graph' is not implemented in mock_mpi"
        )

    def Create_dist_graph_adjacent(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Create_dist_graph_adjacent' is not implemented in mock_mpi"
        )

    def Create_graph(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Create_graph' is not implemented in mock_mpi"
        )

    def Create_group(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Create_group' is not implemented in mock_mpi"
        )

    def Create_intercomm(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Create_intercomm' is not implemented in mock_mpi"
        )

    def Delete_attr(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Delete_attr' is not implemented in mock_mpi"
        )

    def Disconnect(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Disconnect' is not implemented in mock_mpi"
        )

    def Dup(self, *args, **kwargs):
        raise NotImplementedError("The method 'Dup' is not implemented in mock_mpi")

    def Dup_with_info(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Dup_with_info' is not implemented in mock_mpi"
        )

    def Exscan(self, *args, **kwargs):
        raise NotImplementedError("The method 'Exscan' is not implemented in mock_mpi")

    def Free(self, *args, **kwargs):
        raise NotImplementedError("The method 'Free' is not implemented in mock_mpi")

    def Gather(self, *args, **kwargs):
        raise NotImplementedError("The method 'Gather' is not implemented in mock_mpi")

    def Gatherv(self, *args, **kwargs):
        raise NotImplementedError("The method 'Gatherv' is not implemented in mock_mpi")

    def Get_attr(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Get_attr' is not implemented in mock_mpi"
        )

    def Get_errhandler(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Get_errhandler' is not implemented in mock_mpi"
        )

    def Get_group(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Get_group' is not implemented in mock_mpi"
        )

    def Get_info(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Get_info' is not implemented in mock_mpi"
        )

    def Get_name(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Get_name' is not implemented in mock_mpi"
        )

    def Get_topology(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Get_topology' is not implemented in mock_mpi"
        )

    def Graph_map(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Graph_map' is not implemented in mock_mpi"
        )

    def Iallgather(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Iallgather' is not implemented in mock_mpi"
        )

    def Iallgatherv(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Iallgatherv' is not implemented in mock_mpi"
        )

    def Iallreduce(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Iallreduce' is not implemented in mock_mpi"
        )

    def Ialltoall(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Ialltoall' is not implemented in mock_mpi"
        )

    def Ialltoallv(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Ialltoallv' is not implemented in mock_mpi"
        )

    def Ialltoallw(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Ialltoallw' is not implemented in mock_mpi"
        )

    def Ibarrier(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Ibarrier' is not implemented in mock_mpi"
        )

    def Ibcast(self, *args, **kwargs):
        raise NotImplementedError("The method 'Ibcast' is not implemented in mock_mpi")

    def Ibsend(self, *args, **kwargs):
        raise NotImplementedError("The method 'Ibsend' is not implemented in mock_mpi")

    def Idup(self, *args, **kwargs):
        raise NotImplementedError("The method 'Idup' is not implemented in mock_mpi")

    def Iexscan(self, *args, **kwargs):
        raise NotImplementedError("The method 'Iexscan' is not implemented in mock_mpi")

    def Igather(self, *args, **kwargs):
        raise NotImplementedError("The method 'Igather' is not implemented in mock_mpi")

    def Igatherv(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Igatherv' is not implemented in mock_mpi"
        )

    def Improbe(self, *args, **kwargs):
        raise NotImplementedError("The method 'Improbe' is not implemented in mock_mpi")

    def Iprobe(self, *args, **kwargs):
        raise NotImplementedError("The method 'Iprobe' is not implemented in mock_mpi")

    def Irecv(self, *args, **kwargs):
        raise NotImplementedError("The method 'Irecv' is not implemented in mock_mpi")

    def Ireduce(self, *args, **kwargs):
        raise NotImplementedError("The method 'Ireduce' is not implemented in mock_mpi")

    def Ireduce_scatter(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Ireduce_scatter' is not implemented in mock_mpi"
        )

    def Ireduce_scatter_block(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Ireduce_scatter_block' is not implemented in mock_mpi"
        )

    def Irsend(self, *args, **kwargs):
        raise NotImplementedError("The method 'Irsend' is not implemented in mock_mpi")

    def Is_inter(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Is_inter' is not implemented in mock_mpi"
        )

    def Is_intra(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Is_intra' is not implemented in mock_mpi"
        )

    def Iscan(self, *args, **kwargs):
        raise NotImplementedError("The method 'Iscan' is not implemented in mock_mpi")

    def Iscatter(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Iscatter' is not implemented in mock_mpi"
        )

    def Iscatterv(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Iscatterv' is not implemented in mock_mpi"
        )

    def Isend(self, *args, **kwargs):
        raise NotImplementedError("The method 'Isend' is not implemented in mock_mpi")

    def Issend(self, *args, **kwargs):
        raise NotImplementedError("The method 'Issend' is not implemented in mock_mpi")

    def Mprobe(self, *args, **kwargs):
        raise NotImplementedError("The method 'Mprobe' is not implemented in mock_mpi")

    def Probe(self, *args, **kwargs):
        raise NotImplementedError("The method 'Probe' is not implemented in mock_mpi")

    def Recv_init(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Recv_init' is not implemented in mock_mpi"
        )

    def Reduce_scatter(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Reduce_scatter' is not implemented in mock_mpi"
        )

    def Reduce_scatter_block(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Reduce_scatter_block' is not implemented in mock_mpi"
        )

    def Rsend(self, *args, **kwargs):
        raise NotImplementedError("The method 'Rsend' is not implemented in mock_mpi")

    def Rsend_init(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Rsend_init' is not implemented in mock_mpi"
        )

    def Scan(self, *args, **kwargs):
        raise NotImplementedError("The method 'Scan' is not implemented in mock_mpi")

    def Scatter(self, *args, **kwargs):
        raise NotImplementedError("The method 'Scatter' is not implemented in mock_mpi")

    def Scatterv(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Scatterv' is not implemented in mock_mpi"
        )

    def Send_init(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Send_init' is not implemented in mock_mpi"
        )

    def Sendrecv(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Sendrecv' is not implemented in mock_mpi"
        )

    def Sendrecv_replace(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Sendrecv_replace' is not implemented in mock_mpi"
        )

    def Set_attr(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Set_attr' is not implemented in mock_mpi"
        )

    def Set_errhandler(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Set_errhandler' is not implemented in mock_mpi"
        )

    def Set_info(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Set_info' is not implemented in mock_mpi"
        )

    def Set_name(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Set_name' is not implemented in mock_mpi"
        )

    def Spawn(self, *args, **kwargs):
        raise NotImplementedError("The method 'Spawn' is not implemented in mock_mpi")

    def Spawn_multiple(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Spawn_multiple' is not implemented in mock_mpi"
        )

    def Split(self, *args, **kwargs):
        raise NotImplementedError("The method 'Split' is not implemented in mock_mpi")

    def Split_type(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Split_type' is not implemented in mock_mpi"
        )

    def Ssend(self, *args, **kwargs):
        raise NotImplementedError("The method 'Ssend' is not implemented in mock_mpi")

    def Ssend_init(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'Ssend_init' is not implemented in mock_mpi"
        )

    def allgather(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'allgather' is not implemented in mock_mpi"
        )

    def barrier(self, *args, **kwargs):
        raise NotImplementedError("The method 'barrier' is not implemented in mock_mpi")

    def bsend(self, *args, **kwargs):
        raise NotImplementedError("The method 'bsend' is not implemented in mock_mpi")

    def exscan(self, *args, **kwargs):
        raise NotImplementedError("The method 'exscan' is not implemented in mock_mpi")

    def ibsend(self, *args, **kwargs):
        raise NotImplementedError("The method 'ibsend' is not implemented in mock_mpi")

    def improbe(self, *args, **kwargs):
        raise NotImplementedError("The method 'improbe' is not implemented in mock_mpi")

    def iprobe(self, *args, **kwargs):
        raise NotImplementedError("The method 'iprobe' is not implemented in mock_mpi")

    def irecv(self, *args, **kwargs):
        raise NotImplementedError("The method 'irecv' is not implemented in mock_mpi")

    def isend(self, *args, **kwargs):
        raise NotImplementedError("The method 'isend' is not implemented in mock_mpi")

    def issend(self, *args, **kwargs):
        raise NotImplementedError("The method 'issend' is not implemented in mock_mpi")

    def mprobe(self, *args, **kwargs):
        raise NotImplementedError("The method 'mprobe' is not implemented in mock_mpi")

    def probe(self, *args, **kwargs):
        raise NotImplementedError("The method 'probe' is not implemented in mock_mpi")

    def py2f(self, *args, **kwargs):
        raise NotImplementedError("The method 'py2f' is not implemented in mock_mpi")

    def scan(self, *args, **kwargs):
        raise NotImplementedError("The method 'scan' is not implemented in mock_mpi")

    def sendrecv(self, *args, **kwargs):
        raise NotImplementedError(
            "The method 'sendrecv' is not implemented in mock_mpi"
        )

    def ssend(self, *args, **kwargs):
        raise NotImplementedError("The method 'ssend' is not implemented in mock_mpi")

    # Properties not implemented

    @property
    def topology(self):
        raise NotImplementedError(
            "The property 'topology' is not implemented in mock_mpi"
        )

    @property
    def group(self):
        raise NotImplementedError("The property 'group' is not implemented in mock_mpi")

    @property
    def name(self):
        raise NotImplementedError("The property 'name' is not implemented in mock_mpi")

    @property
    def is_inter(self):
        raise NotImplementedError(
            "The property 'is_inter' is not implemented in mock_mpi"
        )

    @property
    def is_intra(self):
        raise NotImplementedError(
            "The property 'is_intra' is not implemented in mock_mpi"
        )

    @property
    def is_topo(self):
        raise NotImplementedError(
            "The property 'is_topo' is not implemented in mock_mpi"
        )

    # 'Info' is the only writeable property
    @property
    def info(self):
        raise NotImplementedError("The property 'info' is not implemented in mock_mpi")

    @info.setter
    def info(self, *args, **kwargs):
        raise NotImplementedError("The property 'info' is not implemented in mock_mpi")

    # Class methods not yet implemented
    @classmethod
    def Compare(cls, *args, **kwargs):
        raise NotImplementedError(
            "The class method 'Compare' is not implemented in mock_mpi"
        )

    @classmethod
    def Get_parent(cls, *args, **kwargs):
        raise NotImplementedError(
            "The class method 'Get_parent' is not implemented in mock_mpi"
        )

    @classmethod
    def Join(cls, *args, **kwargs):
        raise NotImplementedError(
            "The class method 'Join' is not implemented in mock_mpi"
        )

    @classmethod
    def Create_keyval(cls, *args, **kwargs):
        raise NotImplementedError(
            "The class method 'Create_keyval' is not implemented in mock_mpi"
        )

    @classmethod
    def Free_keyval(cls, *args, **kwargs):
        raise NotImplementedError(
            "The class method 'Free_keyval' is not implemented in mock_mpi"
        )

    @classmethod
    def f2py(cls, *args, **kwargs):
        raise NotImplementedError(
            "The class method 'f2py' is not implemented in mock_mpi"
        )
