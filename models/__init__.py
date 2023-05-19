from .GCN_CLF import GCN_CLF
from .GAT_CLF import GAT_CLF


def model_select(name):
    name = name.upper()

    if name in ("GRIDGCN", "GCN"):
        return GCN_CLF
    elif name in ("GRIDGAT", "GAT"):
        return GAT_CLF
    else:
        raise NotImplementedError
