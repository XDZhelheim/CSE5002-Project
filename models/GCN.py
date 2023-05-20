import torch.nn as nn
import torch
import scipy.sparse as sp
import numpy as np
from torchinfo import summary


def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.0
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = (
        sp.eye(adj.shape[0])
        - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    )
    return normalized_laplacian.astype(np.float32).todense()


class GCN(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.W = nn.Parameter(torch.empty(dim_in, dim_out))
        self.b = nn.Parameter(torch.empty(dim_out))
        nn.init.xavier_normal_(self.W)
        nn.init.constant_(self.b, val=0)

    def forward(self, x, adj):
        axw = adj @ x @ self.W + self.b

        return axw


class LapeGCN(nn.Module):
    def __init__(
        self,
        device,
        adj_path,
        num_nodes=5298,
        input_classes_list=[6, 3, 43, 44, 64, 2506],
        embedding_dim=16,
        output_dim=11,
        hidden_dim=32,
        num_layers=3,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.input_classes_list = input_classes_list
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.num_features = len(input_classes_list)

        adj = np.load(adj_path)["data"]
        adj = calculate_normalized_laplacian(adj)
        self.adj = torch.FloatTensor(adj).to(device)

        self.embedding_layers = nn.ModuleList(
            nn.Embedding(num_classes, embedding_dim)
            for num_classes in input_classes_list
        )

        self.input_proj = nn.Linear(self.num_features * embedding_dim, hidden_dim)
        self.gcn_list = nn.ModuleList(
            GCN(dim_in=hidden_dim, dim_out=hidden_dim) for _ in range(num_layers)
        )
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * num_layers, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        # x: (N, C)

        x = x.long()

        embeddings = []
        for i in range(self.num_features):
            embedding = self.embedding_layers[i](x[:, i])  # (N, embedding_dim)
            embeddings.append(embedding)
        x = torch.concat(embeddings, dim=-1)  # (N, embedding_dim*num_features)

        skip_conn = []

        x = self.input_proj(x)  # (N, hidden_dim)
        for gcn in self.gcn_list:
            residual = x
            x = gcn(x, self.adj)  # (N, hidden_dim)
            skip_conn.append(x)
            x += residual

        skip_conn = torch.concat(skip_conn, dim=-1)

        out = self.output_proj(skip_conn)  # (N, output_dim)

        return out


class DFGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, dropout=0.1):
        super(DFGCN, self).__init__()
        self.cheb_k = cheb_k
        self.dim_in = dim_in
        self.W = nn.Parameter(torch.empty(cheb_k * dim_in, dim_out), requires_grad=True)
        self.b = nn.Parameter(torch.empty(dim_out), requires_grad=True)
        nn.init.xavier_normal_(self.W)
        nn.init.constant_(self.b, val=0)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, G):
        """
        :param x: graph feature/signal          -   [N, C]
        :param G: support adj matrices          -   [K, N, N]
        :return output: hidden representation   -   [N, H_out]
        """
        support_list = []
        for k in range(self.cheb_k):
            support = torch.matmul(G[k, :, :], x)  # [N, C] perform GCN
            support = self.dropout(support)
            support_list.append(support)  # k * [N, C]
        support_cat = torch.cat(support_list, dim=-1)  # [N, k * C]
        output = torch.matmul(support_cat, self.W) + self.b  # [N, H_out]

        return output


class ADFGCN(nn.Module):
    def __init__(
        self,
        device,
        adj_path,
        num_nodes=5298,
        input_classes_list=[6, 3, 43, 44, 64, 2506],
        embedding_dim=16,
        output_dim=11,
        hidden_dim=64,
        node_embedding_dim=16,
        cheb_k=1,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.input_classes_list = input_classes_list
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.node_embedding_dim = node_embedding_dim
        self.cheb_k = cheb_k
        self.num_layers = num_layers

        self.num_features = len(input_classes_list)

        adj = np.load(adj_path)["data"]
        adj = [asym_adj(adj), asym_adj(np.transpose(adj))]
        self.P = self.compute_cheby_poly(adj).to(device)
        k = self.P.shape[0]

        if node_embedding_dim > 0:
            self.node_emb1 = nn.Parameter(torch.randn(num_nodes, node_embedding_dim))
            self.node_emb2 = nn.Parameter(torch.randn(num_nodes, node_embedding_dim))
            k += cheb_k

        self.embedding_layers = nn.ModuleList(
            nn.Embedding(num_classes, embedding_dim)
            for num_classes in input_classes_list
        )

        self.input_proj = nn.Linear(self.num_features * embedding_dim, hidden_dim)
        self.gcn_list = nn.ModuleList(
            DFGCN(dim_in=hidden_dim, dim_out=hidden_dim, cheb_k=k, dropout=dropout)
            for _ in range(num_layers)
        )
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * num_layers, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        # x: (N, C)

        # assert self.num_features == x.shape[1]
        x = x.long()

        supports = [self.P]

        if self.node_embedding_dim > 0:
            adp = self.node_emb1 @ self.node_emb2.T
            adp = torch.softmax(torch.relu(adp), dim=-1)  # (N, N)
            adps = [adp]
            for _ in range(self.cheb_k - 1):
                adp = adp @ adp
                adps.append(adp)
            adps = torch.stack(adps)  # (K, N, N)
            supports.append(adps)

        supports = torch.concat(supports, dim=0)

        embeddings = []
        for i in range(self.num_features):
            embedding = self.embedding_layers[i](x[:, i])  # (N, embedding_dim)
            embeddings.append(embedding)
        x = torch.concat(embeddings, dim=-1)  # (N, embedding_dim*num_features)

        skip_conn = []

        x = self.input_proj(x)  # (N, hidden_dim)
        for gcn in self.gcn_list:
            residual = x
            x = gcn(x, supports)  # (N, hidden_dim)
            skip_conn.append(x)
            x += residual

        skip_conn = torch.concat(skip_conn, dim=-1)

        out = self.output_proj(skip_conn)  # (N, output_dim)

        # out = torch.softmax(out, dim=-1)

        return out

    def compute_cheby_poly(self, P: list):
        P_k = []
        for p in P:
            p = torch.from_numpy(p).float().T
            T_k = [torch.eye(p.shape[0]), p]  # order 0, 1
            for k in range(2, self.cheb_k):
                T_k.append(2 * torch.mm(p, T_k[-1]) - T_k[-2])  # recurrent to order K
            P_k += T_k
        return torch.stack(P_k, dim=0)  # (K, N, N) or (2*K, N, N) for bidirection


if __name__ == "__main__":
    model = ADFGCN(
        device=torch.device("cpu"),
        adj_path="../data/adj.npz",
        num_layers=2,
        node_embedding_dim=16,
    )
    summary(model, [5298, 6], device="cpu")
