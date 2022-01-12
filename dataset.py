import math
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import dense_to_sparse


class ConnectivityData(InMemoryDataset):
    """ Dataset for the connectivity data."""

    def __init__(self,
                 root,
                 sparsity=.2):
        self.sparsity = sparsity
        super(ConnectivityData, self).__init__(root, None, None)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        file_paths = sorted(list(Path(self.raw_dir).glob("*.txt")))
        return [str(file_path.name) for file_path in file_paths]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def binarize(self, matrix):
        """ Calculate the adjacency matrix from the connectivity matrix."""
        triu_numel = int((matrix.size - np.diag(matrix).size) / 2)
        threshold = math.floor(self.sparsity * triu_numel)
        sorted_idx = np.argsort(np.triu(np.abs(matrix), 1), axis=None)
        valid_idx = sorted_idx[-threshold:]
        adj_mat = np.zeros(matrix.shape, dtype='int32').flatten()
        adj_mat[valid_idx] = 1
        adj_mat = adj_mat.reshape(matrix.shape)
        return adj_mat + adj_mat.transpose()

    def process(self):
        labels = np.genfromtxt(Path(self.raw_dir) / "Labels.csv")

        data_list = []
        for filename, y in zip(self.raw_paths, labels):
            y = torch.tensor([y]).long()
            connectivity = np.genfromtxt(filename)
            x = torch.from_numpy(connectivity).float()

            adj = self.binarize(connectivity)
            adj = torch.from_numpy(adj).float()
            edge_index = dense_to_sparse(adj)[0]

            data_list.append(Data(x=x, edge_index=edge_index, y=y))

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
