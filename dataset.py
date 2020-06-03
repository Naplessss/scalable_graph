import torch 
import math
import numpy as np
import torch.distributed as dist

from torch_geometric.data import Data, ClusterData, ClusterLoader, NeighborSampler
from torch.utils.data import DataLoader, TensorDataset, IterableDataset
 
  
class NeighborSampleDataset(IterableDataset):
    def __init__(self, X, y, edge_index, edge_weight, num_nodes, batch_size, shuffle=False, use_dist_sampler=False):
        self.X = X
        self.y = y

        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.num_nodes = num_nodes
        self.batch_size = batch_size
        self.shuffle = shuffle
        # whether to use distributed sampler when available
        self.use_dist_sampler = use_dist_sampler
        # use 'epoch' as the random seed to shuffle data for distributed training
        self.epoch = None

        self.graph_sampler = self._make_graph_sampler()
        self.length = self.get_length()

    def _make_graph_sampler(self):
        graph = Data(
            edge_index=self.edge_index, edge_attr=self.edge_weight, num_nodes=self.num_nodes
        ).to('cpu')

        graph_sampler = NeighborSampler(
            # graph, size=[5, 5], num_hops=2, batch_size=100, shuffle=self.shuffle, add_self_loops=True
            graph, size=[10, 15], num_hops=2, batch_size=250, shuffle=self.shuffle, add_self_loops=True
        )

        return graph_sampler

    def get_subgraph(self, data_flow):
        sub_graph = {
            'edge_index': [block.edge_index for block in data_flow],
            'edge_weight': [self.edge_weight[block.e_id] for block in data_flow],
            'size': [block.size for block in data_flow],
            'res_n_id': [block.res_n_id for block in data_flow],
            'cent_n_id': data_flow[-1].n_id[data_flow[-1].res_n_id],
            'graph_n_id': data_flow[0].n_id
        }
        return sub_graph

    def __iter__(self):
        if self.use_dist_sampler and dist.is_initialized():
            # ensure that all processes share the same graph dataflow
            torch.manual_seed(self.epoch)

        for data_flow in self.graph_sampler():
            g = self.get_subgraph(data_flow)
            X, y = self.X[:, g['graph_n_id']], self.y[:, g['cent_n_id']]
            dataset_len = X.size(0)
            indices = list(range(dataset_len))

            if self.use_dist_sampler and dist.is_initialized():
                # distributed sampler reference: torch.utils.data.distributed.DistributedSampler
                if self.shuffle:
                    # ensure that all processes share the same permutated indices
                    tg = torch.Generator()
                    tg.manual_seed(self.epoch)
                    indices = torch.randperm(dataset_len, generator=tg).tolist()

                world_size = dist.get_world_size()
                node_rank = dist.get_rank()
                num_samples_per_node = int(math.ceil(dataset_len * 1.0 / world_size))
                total_size = world_size * num_samples_per_node

                # add extra samples to make it evenly divisible
                indices += indices[:(total_size - dataset_len)]
                assert len(indices) == total_size

                # get sub-batch for each process
                # Node (rank=x) get [x, x+world_size, x+2*world_size, ...]
                indices = indices[node_rank:total_size:world_size]
                assert len(indices) == num_samples_per_node
            elif self.shuffle:
                np.random.shuffle(indices)

            num_batches = (len(indices) + self.batch_size - 1) // self.batch_size
            for batch_id in range(num_batches):
                start = batch_id * self.batch_size
                end = (batch_id + 1) * self.batch_size
                yield X[indices[start: end]], y[indices[start: end]], g, torch.LongTensor(indices[start: end])

    def get_length(self):
        length = 0
        for data_flow in self.graph_sampler():
            if self.use_dist_sampler and dist.is_initialized():
                dataset_len = self.X.size(0)
                world_size = dist.get_world_size()
                num_samples_per_node = int(math.ceil(dataset_len * 1.0 / world_size))
            else:
                num_samples_per_node = self.X.size(0)
            length += (num_samples_per_node + self.batch_size - 1) // self.batch_size

        return length

    def __len__(self):
        return self.length

    def set_epoch(self, epoch):
        # self.set_epoch() will be called by BasePytorchTask on each epoch when using distributed training
        self.epoch = epoch

class ClusterDataset(IterableDataset):
    def __init__(self, X, y, edge_index, edge_weight, num_nodes, batch_size, shuffle=False, use_dist_sampler=False):
        self.X = X
        self.y = y

        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.num_nodes = num_nodes
        self.batch_size = batch_size
        self.shuffle = shuffle
        # whether to use distributed sampler when available
        self.use_dist_sampler = use_dist_sampler
        # use 'epoch' as the random seed to shuffle data for distributed training
        self.epoch = None

        self.graph_sampler = self._make_graph_sampler()
        self.length = self.get_length()

    def _make_graph_sampler(self):
        graph = Data(
            edge_index=self.edge_index, edge_attr=self.edge_weight, 
            n_id=torch.arange(0, self.num_nodes), num_nodes=self.num_nodes
        ).to('cpu')

        cluster_data = ClusterData(
            graph, num_parts=100, recursive=False, save_dir=None
        )

        cluster_loader = ClusterLoader(cluster_data, batch_size=5, shuffle=True, num_workers=0)

        return cluster_loader

    def get_subgraph(self, subgraph):
        graph = dict()

        device = self.edge_index.device

        graph['edge_index'] = subgraph.edge_index.to(device)
        graph['edge_weight'] = subgraph.edge_attr.to(device)
        graph['cent_n_id'] = subgraph.n_id

        return graph

    def __iter__(self):
        if self.use_dist_sampler and dist.is_initialized():
            # ensure that all processes share the same graph dataflow
            torch.manual_seed(self.epoch)

        for subgraph in self.graph_sampler:
            g = self.get_subgraph(subgraph)
            X, y = self.X[:, g['cent_n_id']], self.y[:, g['cent_n_id']]
            dataset_len = X.size(0)
            indices = list(range(dataset_len))

            if self.use_dist_sampler and dist.is_initialized():
                # distributed sampler reference: torch.utils.data.distributed.DistributedSampler
                if self.shuffle:
                    # ensure that all processes share the same permutated indices
                    tg = torch.Generator()
                    tg.manual_seed(self.epoch)
                    indices = torch.randperm(dataset_len, generator=tg).tolist()

                world_size = dist.get_world_size()
                node_rank = dist.get_rank()
                num_samples_per_node = int(math.ceil(dataset_len * 1.0 / world_size))
                total_size = world_size * num_samples_per_node

                # add extra samples to make it evenly divisible
                indices += indices[:(total_size - dataset_len)]
                assert len(indices) == total_size

                # get sub-batch for each process
                # Node (rank=x) get [x, x+world_size, x+2*world_size, ...]
                indices = indices[node_rank:total_size:world_size]
                assert len(indices) == num_samples_per_node
            elif self.shuffle:
                np.random.shuffle(indices)

            num_batches = (len(indices) + self.batch_size - 1) // self.batch_size
            for batch_id in range(num_batches):
                start = batch_id * self.batch_size
                end = (batch_id + 1) * self.batch_size
                yield X[indices[start: end]], y[indices[start: end]], g, torch.LongTensor(indices[start: end])

    def get_length(self):
        length = 0
        for _ in self.graph_sampler:
            if self.use_dist_sampler and dist.is_initialized():
                dataset_len = self.X.size(0)
                world_size = dist.get_world_size()
                num_samples_per_node = int(math.ceil(dataset_len * 1.0 / world_size))
            else:
                num_samples_per_node = self.X.size(0)
            length += (num_samples_per_node + self.batch_size - 1) // self.batch_size

        return length

    def __len__(self):
        return self.length

    def set_epoch(self, epoch):
        # self.set_epoch() will be called by BasePytorchTask on each epoch when using distributed training
        self.epoch = epoch 