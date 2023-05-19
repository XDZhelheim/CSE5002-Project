from .GCN_CLF import GCN_CLF
from .GAT_CLF import GAT_CLF
from .MLP import MLP


def model_select(name):
    name = name.upper()

    if name in ("GCN",):
        return GCN_CLF
    elif name in ("GAT",):
        return GAT_CLF
    elif name == "MLP":
        return MLP
    else:
        raise NotImplementedError
