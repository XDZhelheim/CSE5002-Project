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


class GCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k):
        super(GCN, self).__init__()
        self.cheb_k = cheb_k
        self.dim_in = dim_in
        self.W = nn.Parameter(torch.empty(cheb_k * dim_in, dim_out), requires_grad=True)
        self.b = nn.Parameter(torch.empty(dim_out), requires_grad=True)
        nn.init.xavier_normal_(self.W)
        nn.init.constant_(self.b, val=0)

    def forward(self, x, G):
        """
        :param x: graph feature/signal          -   [B, N, C]
        :param G: support adj matrices          -   [K, N, N]
        :return output: hidden representation   -   [B, N, H_out]
        """
        support_list = []
        for k in range(self.cheb_k):
            support = torch.einsum(
                "ij,bjp->bip", G[k, :, :], x
            )  # [B, N, C] perform GCN
            support_list.append(support)  # k * [B, N, C]
        support_cat = torch.cat(support_list, dim=-1)  # [B, N, k * C]
        output = (
            torch.einsum("bip,pq->biq", support_cat, self.W) + self.b
        )  # [B, N, H_out]
        return output


class GridGCN(nn.Module):
    def __init__(
        self,
        device,
        num_grids_height=20,
        num_grids_width=20,
        input_dim=3,
        output_dim=1,
        hidden_dim=32,
        grid_embedding_dim=16,
        cheb_k=3,
        num_layers=3,
    ):
        super().__init__()

        self.num_grids_height = num_grids_height
        self.num_grids_width = num_grids_width
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.grid_embedding_dim = grid_embedding_dim
        self.cheb_k = cheb_k
        self.num_layers = num_layers

        adj = self.gen_adj(num_grids_height, num_grids_width)
        adj = [asym_adj(adj), asym_adj(np.transpose(adj))]
        self.P = self.compute_cheby_poly(adj).to(device)
        k = self.P.shape[0]

        if grid_embedding_dim > 0:
            self.grid_emb1 = nn.Parameter(
                torch.randn(num_grids_height * num_grids_width, grid_embedding_dim)
            )
            self.grid_emb2 = nn.Parameter(
                torch.randn(num_grids_height * num_grids_width, grid_embedding_dim)
            )
            k += cheb_k

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.gcn_list = nn.ModuleList(
            GCN(dim_in=hidden_dim, dim_out=hidden_dim, cheb_k=k)
            for _ in range(num_layers)
        )
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        # x: (batch_size, num_grids_height, num_grids_width, 2)
        batch_size = x.shape[0]
        x = x.view(
            batch_size, self.num_grids_height * self.num_grids_width, self.input_dim
        )

        supports = [self.P]

        if self.grid_embedding_dim > 0:
            adp = self.grid_emb1 @ self.grid_emb2.T
            adp = torch.softmax(torch.relu(adp), dim=-1)  # (H*W, H*W)
            adps = [adp]
            for _ in range(self.cheb_k - 1):
                adp = adp @ adp
                adps.append(adp)
            adps = torch.stack(adps)  # (K, H*W, H*W)
            supports.append(adps)

        supports = torch.concat(supports, dim=0)

        x = self.input_proj(x)  # (B, H*W, hidden_dim)
        for gcn in self.gcn_list:
            x = gcn(x, supports)  # (B, H*W, hidden_dim)
        out = self.output_proj(x)  # (B, H*W, output_dim)

        out = out.view(
            batch_size, self.num_grids_height, self.num_grids_width, self.output_dim
        )

        return out

    def gen_adj(self, num_grids_height=20, num_grids_width=20):
        """
        Returns:
        N*N 01 adj matrix,
        N=num_h*num_w
        ```
        0 1 2
        3 4 5
        6 7 8

        1 1 0 1 0 0 0 0 0
        1 1 1 0 1 0 0 0 0
        0 1 1 0 0 1 0 0 0
        1 0 0 1 1 0 1 0 0
        0 1 0 1 1 1 0 1 0
        0 0 1 0 1 1 0 0 1
        0 0 0 1 0 0 1 1 0
        0 0 0 0 1 0 1 1 1
        0 0 0 0 0 1 0 1 1
        ```
        """
        adj = np.zeros(
            (num_grids_height, num_grids_width, num_grids_height, num_grids_width)
        )
        for i in range(num_grids_height):
            for j in range(num_grids_width):
                if i - 1 >= 0:
                    # top
                    adj[i - 1, j, i, j] = 1
                    adj[i, j, i - 1, j] = 1
                if i + 1 < num_grids_height:
                    # bottom
                    adj[i + 1, j, i, j] = 1
                    adj[i, j, i + 1, j] = 1
                if j - 1 >= 0:
                    # left
                    adj[i, j - 1, i, j] = 1
                    adj[i, j, i, j - 1] = 1
                if j + 1 < num_grids_width:
                    # right
                    adj[i, j + 1, i, j] = 1
                    adj[i, j, i, j + 1] = 1

        adj = adj.reshape(
            num_grids_height * num_grids_width, num_grids_height * num_grids_width
        )

        for i in range(num_grids_height * num_grids_width):
            adj[i, i] = 1

        return adj

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
    model = GridGCN(device=torch.device("cpu"), num_layers=3, grid_embedding_dim=16)
    summary(model, [64, 20, 20, 3], device="cpu")
    # print(model.gen_adj(3, 3))
