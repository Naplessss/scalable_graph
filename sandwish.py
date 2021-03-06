import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from gcn import SAGENet, GATNet, ClusterSAGENet, ClusterGATNet
from egnn import SAGELANet, ClusterSAGELANet, GatedGCNNet, ClusterGatedGCNNet, MyEGNNNet, ClusterMyEGNNNet
from krnn import KRNN

from torch_geometric.data import Data, Batch, DataLoader, NeighborSampler, ClusterData, ClusterLoader


class GCNBlock(nn.Module):
    def __init__(self, in_channels, spatial_channels, edge_channels, skip_connection, num_nodes,
                 gcn_type, gcn_partition):
        super(GCNBlock, self).__init__()
        if gcn_partition == 'cluster':
            GCNUnit = {'cluster_sagela': ClusterSAGELANet, 
                        'cluster_gated': ClusterGatedGCNNet,
                        'cluster_sage': ClusterSAGENet,
                        'cluster_gat': ClusterGATNet,
                        'cluster_my': ClusterMyEGNNNet}.get(gcn_type)
        elif gcn_partition == 'sample':
            GCNUnit = {'sage': SAGENet, 
                        'gat': GATNet, 
                        'sagela': SAGELANet, 
                        'gated': GatedGCNNet,
                        'my': MyEGNNNet}.get(gcn_type)
        if gcn_type in ['gated','my','sagela','cluster_sagela','cluster_gated','cluster_my']:
            self.gcn = GCNUnit(in_channels=in_channels,
                                out_channels=spatial_channels,
                                edge_channels=edge_channels,
                                skip_connection=skip_connection)
        else:
            self.gcn = GCNUnit(in_channels=in_channels,
                                out_channels=spatial_channels,
                                skip_connection=skip_connection)           

    def forward(self, X, g):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param g: graph information.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        t1 = X.permute(0, 2, 1, 3).contiguous(
        ).view(-1, X.shape[1], X.shape[3])
        t2 = F.relu(self.gcn(t1, g))
        out = t2.view(X.shape[0], X.shape[2], t2.shape[1],
                      t2.shape[2]).permute(0, 2, 1, 3)

        return out


class Sandwich(nn.Module):
    def __init__(self, num_nodes, num_edges, num_features, 
                 num_timesteps_input, num_timesteps_output, num_edge_features=1,skip_connection=False,
                 gcn_type='sage', gcn_partition='sample', hidden_size=64, use_gcn=True, **kwargs):
        """
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(Sandwich, self).__init__()
        self.gcn_partition = gcn_partition
        self.use_gcn = use_gcn
        self.gru1 = KRNN(num_nodes, num_features, num_timesteps_input, num_timesteps_output=None, hidden_size=hidden_size)

        if self.use_gcn:
            self.gcn = GCNBlock(in_channels=hidden_size,
                            spatial_channels=hidden_size,
                            edge_channels=num_edge_features,
                            skip_connection=skip_connection,
                            num_nodes=num_nodes,
                            gcn_type=gcn_type,
                            gcn_partition=gcn_partition)

        self.gru = KRNN(num_nodes, hidden_size, num_timesteps_input,
                        num_timesteps_output, hidden_size)

    def forward(self, X, g):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        """
        if self.gcn_partition == 'sample':
            out1 = self.gru1(X, g['graph_n_id'])
        else:
            out1 = self.gru1(X, g['cent_n_id'])

        if self.use_gcn:
            out1 = self.gcn(out1, g)
        if not self.use_gcn and self.gcn_partition == 'sample':
            out1 = out1[:,g['res_n_id'][-1]]
        out2 = self.gru(out1, g['cent_n_id'])
        # out2 = out2.squeeze(dim=-1)       
        return out2