from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_networkx

import torch
import numpy as np
import networkx as nx

import pickle as pkl
from random import shuffle

def community_small(num_communities, comm_min_size, comm_max_size, p_interconnect=0.01):
    c_sizes = np.random.choice(np.arange(comm_min_size, comm_max_size+1), num_communities)
    graphs = [nx.gnp_random_graph(c_sizes[i], 0.7, seed=i) for i in range(len(c_sizes))]
    G = nx.disjoint_union_all(graphs)

    communities = list(nx.connected_components(G))
    for i in range(len(communities)):
        nodes1 = list(communities[i])
        for j in range(i+1, len(communities)):
            nodes2 = list(communities[j])
            has_inter_edge = False
            for n1 in nodes1:
                for n2 in nodes2:
                    if np.random.rand() < p_interconnect:
                        G.add_edge(n1, n2)
                        has_inter_edge = True
            if not has_inter_edge:
                G.add_edge(np.random.choice(nodes1, 1).item(),
                        np.random.choice(nodes2, 1).item())

    return G


class GNFCommunitySmall(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['community_small.pt']

    def process(self):
        data_list = [
            from_networkx(
                community_small(
                    num_communities=2,
                    comm_min_size=6,
                    comm_max_size=10)
            ) for _ in range(100)
        ]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class GNFEgoSmall(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['ind.citeseer.graph']

    @property
    def processed_file_names(self):
        return ['ego_small.pt']

    def process(self):
        graph = pkl.load(open(self.raw_paths[0], 'rb'), encoding='latin1')
        G = nx.from_dict_of_lists(graph)
        G.remove_edges_from(nx.selfloop_edges(G))

        cc = max(nx.connected_components(G), key=len)
        G.subgraph(cc)
        G = nx.convert_node_labels_to_integers(G)

        graphs = []
        for i in range(G.number_of_nodes()):
            G_ego = nx.ego_graph(G, i, radius=1)
            if (G_ego.number_of_nodes() >= 4) and (G_ego.number_of_nodes() <= 20):
                graphs.append(G_ego)
        shuffle(graphs)
        graphs = graphs[0:200]

        data_list = [from_networkx(g) for g in graphs]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
