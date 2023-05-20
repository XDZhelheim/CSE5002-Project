from .GCN import ADFGCN, LapeGCN
from .GAT_CLF import GAT_CLF
from .MLP import MLP


def model_select(name):
    name = name.upper()

    if name == "ADFGCN":
        return ADFGCN
    elif name in ("GCN", "LAPEGCN"):
        return LapeGCN
    elif name == "GAT":
        return GAT_CLF
    elif name == "MLP":
        return MLP
    else:
        raise NotImplementedError
