from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch

class MLPDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_channels, 2 * hidden_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(2 * hidden_channels, 2 * hidden_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(2 * hidden_channels, 1)
        )

    def forward(self, z, edge_index, sigmoid=True):
        x = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
        x = self.mlp(x).ravel()

        if sigmoid:
            x = torch.sigmoid(x)

        return x

    def forward_all(self, z, sigmoid=True):
        num_vert = len(z)

        n_arange = torch.arange(num_vert)
        grid_x, grid_y = torch.meshgrid(n_arange, n_arange)
        grid_x, grid_y = grid_x.ravel(), grid_y.ravel()

        x = torch.cat([z[grid_x], z[grid_y]], dim=1)
        x = self.mlp(x).ravel().reshape(num_vert, num_vert)

        if sigmoid:
            x = torch.sigmoid(x)

        return x


class MLPInnerProductDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.SiLU(inplace=True),
            torch.nn.Linear(hidden_channels, hidden_channels),
            # torch.nn.Tanh()
        )

    def forward(self, z, edge_index, sigmoid=True):
        z = self.mlp(z)

        x = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)

        if sigmoid:
            x = torch.sigmoid(x)

        return x

    def forward_all(self, z, sigmoid=True):
        z = self.mlp(z)

        adj = torch.matmul(z, z.t())

        if sigmoid:
            adj = torch.sigmoid(adj)

        return adj


        

class InnerProductDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z, edge_index, sigmoid=True):
        x = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)

        if sigmoid:
            x = torch.sigmoid(x)

        return x

    def forward_all(self, z, sigmoid=True):
        adj = torch.matmul(z, z.t())

        if sigmoid:
            adj = torch.sigmoid(adj)

        return adj



class InnerProductDecoderBasis(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()

        basis = torch.randn((hidden_channels, hidden_channels)).cuda()
        self.inner_prod_matrix = torch.matmul(basis.t(), basis)


    def forward(self, z, edge_index, sigmoid=True):
        z1 = z[edge_index[0]]
        z2 = torch.matmul(self.inner_prod_matrix, z[edge_index[1]].t()).t()

        x = (z1 * z2).sum(dim=1)

        if sigmoid:
            x = torch.sigmoid(x)

        return x


    def forward_all(self, z, sigmoid=True):
        adj = torch.matmul(z, torch.matmul(self.inner_prod_matrix, z.t()))

        if sigmoid:
            adj = torch.sigmoid(adj)

        return adj

    

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(1, num_layers-1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.conv_mu = GCNConv(hidden_channels, out_channels)
        self.conv_logstd = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        for i in range(self.num_layers-1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)

        mu = self.conv_mu(x, edge_index)
        logstd = self.conv_logstd(x, edge_index)

        return mu, logstd


class VGAE(torch.nn.Module):
    def __init__(self, encoder, decoder,
                       lambda1, lambda2):
        super().__init__()

        self.lambda1 = 2 * lambda1 / (lambda1 + lambda2)
        self.lambda2 = 2 * lambda2 / (lambda1 + lambda2)

        self.encoder = encoder
        self.decoder = decoder
        self.bce_loss = torch.nn.BCELoss(reduction='mean')

    def recon_loss(self, y_pred, y_true):
        return self.lambda1 * self.bce_loss(y_pred, y_true).mean()

    def dkl_loss(self, mu, logstd):
        return self.lambda2 * -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))

    def reparametrize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def encode(self, x, edge_index):
        mu, logstd = self.encoder(x, edge_index)
        z = self.reparametrize(mu, logstd)

        return z, mu, logstd

    def decode(self, z, edge_index=None):
        if edge_index is not None:
            return self.decoder(z, edge_index)
        
        return self.decoder.forward_all(z)
