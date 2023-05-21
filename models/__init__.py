from .GCN import ADFGCN, LapeGCN
from .GAT import GAT
from .MLP import MLP


def model_select(name):
    name = name.upper()

    if name == "ADFGCN":
        return ADFGCN
    elif name in ("GCN", "LAPEGCN"):
        return LapeGCN
    elif name == "GAT":
        return GAT
    elif name == "MLP":
        return MLP
    else:
        raise NotImplementedError
