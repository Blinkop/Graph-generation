import torch_geometric.transforms as T
import torch
from torch_geometric.utils import negative_sampling

class RandomFeatureMatrix(T.BaseTransform):
    def __init__(self, feature_dim, mean, std):
        self.feature_dim = feature_dim
        self.mean = mean
        self.std = std

    def __call__(self, data):
        x = torch.normal(
            mean=self.mean,
            std=self.std,
            size=(data.num_nodes, self.feature_dim)
        )
        data['x'] = x

        return data

    def __repr__(self):
        return f'{self.__class__.__name__}({self.feature_dim})'


class OneHotFeatureMatrix(T.BaseTransform):
    def __init__(self, max_num_nodes):
        self.max_num_nodes = max_num_nodes

    def __call__(self, data):
        x = torch.zeros(data.num_nodes, self.max_num_nodes)
        x[:, :data.num_nodes] = torch.eye(data.num_nodes)

        data['x'] = x

        return data

    def __repr__(self):
        return f'{self.__class__.__name__}({self.feature_dim})'


# class EdgeLabelCreator(T.RandomLinkSplit):
#     def __init__(self, is_undirected):
#         super().__init__(0.0, 0.0, is_undirected=is_undirected)

#     def __call__(self, data):
#         return super().__call__(data)[0]

#     def __repr__(self):
#         return f'{self.__class__.__name__}({self.is_undirected})'


class EdgeLabelCreator(T.BaseTransform):
    def __init__(self, is_undirected):
        self.is_undirected = is_undirected

    def __call__(self, data):
        neg_edge_index = negative_sampling(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            force_undirected=self.is_undirected
        )

        perm = torch.randperm(data.edge_index.shape[1])
        edge_index = data.edge_index[:, perm]

        data['edge_label'] = torch.cat([
            torch.ones(edge_index.shape[1]),
            torch.zeros(neg_edge_index.shape[1])
        ])

        data['edge_label_index'] = torch.cat([
            edge_index,
            neg_edge_index], dim=1
        )

        return data

    def __repr__(self):
        return f'{self.__class__.__name__}({self.is_undirected})'


class FullEdgeLabelCreator(T.BaseTransform):
    def __init__(self):
        pass

    def __call__(self, data):
        adj = torch.zeros(data.num_nodes, data.num_nodes)
        adj[data.edge_index[0], data.edge_index[1]] = 1

        data['edge_label'] = adj.ravel()

        n_arange = torch.arange(data.num_nodes)
        grid_x, grid_y = torch.meshgrid(n_arange, n_arange)
        grid_x, grid_y = grid_x.ravel(), grid_y.ravel()

        data['edge_label_index'] = torch.stack([grid_x, grid_y])

        return data

    def __repr__(self):
        return f'{self.__class__.__name__}'
