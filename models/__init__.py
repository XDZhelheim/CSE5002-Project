from .GridGCN import GridGCN
from .GridGAT import GridGAT


def model_select(name):
    name = name.upper()

    if name in ("GRIDGCN", "GCN"):
        return GridGCN
    elif name in ("GRIDGAT", "GAT"):
        return GridGAT
    else:
        raise NotImplementedError
