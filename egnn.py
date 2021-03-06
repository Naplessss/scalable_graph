import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as PyG
from torch_geometric.nn.conv import MessagePassing


class SAGELA(PyG.SAGEConv):
    """
    The SAGE-LA operator from the `"GCN-LASE: Towards Adequately 
    Incorporating Link Attributes in Graph Convolutional Networks" 
    <https://arxiv.org/abs/1902.09817>`_ paper
    """
    def __init__(self, in_channels, out_channels, edge_channels,
                 normalize=False, concat=True, bias=True, **kwargs):
        super(SAGELA, self).__init__(in_channels, out_channels, normalize=normalize,
                                     concat=concat, bias=bias, **kwargs)

        self.edge_channels = edge_channels
        self.amp_weight = nn.Parameter(torch.Tensor(edge_channels, in_channels))
        self.gate_linear = nn.Linear(2 * in_channels + edge_channels, 1)
        nn.init.xavier_uniform_(self.amp_weight)

    def forward(self, x, edge_index, edge_feature, size=None,
                res_n_id=None):
        return self.propagate(edge_index, size=size, x=x,
                              edge_feature=edge_feature, res_n_id=res_n_id)

    def message(self, x_i, x_j, edge_feature):
        # calculate gate lambda
        lamb_in = torch.cat([x_i, x_j, edge_feature.repeat(x_j.shape[0], 1, 1)], dim=-1)
        lamb = torch.sigmoid(self.gate_linear(lamb_in))

        # amplifiew
        amp = torch.matmul(edge_feature, self.amp_weight)
        amp_x_j = amp * x_j

        return amp_x_j * lamb

    def update(self, aggr_out, x):
        if self.concat and torch.is_tensor(x):
            aggr_out = torch.cat([x, aggr_out], dim=-1)
        elif self.concat and (isinstance(x, tuple) or isinstance(x, list)):
            aggr_out = torch.cat([x[1], aggr_out], dim=-1)

        aggr_out = torch.matmul(aggr_out, self.weight)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias

        if self.normalize:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)

        return aggr_out


class SAGELANet(nn.Module):
    def __init__(self, in_channels, out_channels, edge_channels=1, spatial_channels=16, skip_connection=False):
        super(SAGELANet, self).__init__()
        self.skip_connection = skip_connection
        conv2_in_channels = spatial_channels * 2 if self.skip_connection else spatial_channels
        self.conv1 = SAGELA(
            in_channels, spatial_channels, edge_channels=edge_channels, node_dim=1)
        self.conv2 = SAGELA(
            conv2_in_channels, out_channels, edge_channels=edge_channels, node_dim=1)

    def forward(self, X, g):
        edge_index = g['edge_index']
        edge_weight = g['edge_weight']

        size = g['size']
        res_n_id = g['res_n_id']

        X = self.conv1(
            (X, X[:, res_n_id[0]]), edge_index[0], edge_feature=edge_weight[0], size=size[0])

        if self.skip_connection:
            X = torch.cat((F.leaky_relu(X), X), dim=-1)
        else:
            X = F.leaky_relu(X)

        X = self.conv2(
            (X, X[:, res_n_id[1]]), edge_index[1], edge_feature=edge_weight[1], size=size[1])

        return X


class ClusterSAGELANet(nn.Module):
    def __init__(self, in_channels, out_channels, edge_channels=1, spatial_channels=16, skip_connection=False):
        super(ClusterSAGELANet, self).__init__()
        self.skip_connection = skip_connection
        conv2_in_channels = spatial_channels * 2 if self.skip_connection else spatial_channels
        self.conv1 = SAGELA(
            in_channels, spatial_channels, edge_channels=edge_channels, node_dim=1)
        self.conv2 = SAGELA(
            conv2_in_channels, out_channels, edge_channels=edge_channels, node_dim=1)

    def forward(self, X, g):
        edge_index = g['edge_index']
        edge_weight = g['edge_weight']

        X = self.conv1(X, edge_index, edge_feature=edge_weight)

        if self.skip_connection:
            X = torch.cat((F.leaky_relu(X), X), dim=-1)
        else:
            X = F.leaky_relu(X)  

        X = self.conv2(X, edge_index, edge_feature=edge_weight)

        return X


class GatedGCN(MessagePassing):
    """
    The GatedGCN operator from the `"Residual Gated Graph ConvNets" 
    <https://arxiv.org/abs/1711.07553>`_ paper
    """
    def __init__(self, in_channels, out_channels, edge_channels, 
                 **kwargs):
        super(GatedGCN, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_channels = edge_channels
        
        self.weight1 = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.weight2 = nn.Parameter(torch.Tensor(edge_channels, out_channels))

        self.u = nn.Parameter(torch.Tensor(out_channels, out_channels))
        self.v = nn.Parameter(torch.Tensor(out_channels, out_channels))

        # self.batch_norm = nn.BatchNorm1d(num_features=out_channels)
        # self.layer_norm = nn.LayerNorm(normalized_shape=out_channels)

        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight1)
        nn.init.xavier_uniform_(self.weight2)
        nn.init.xavier_uniform_(self.u)
        nn.init.xavier_uniform_(self.v)

    def forward(self, x, edge_index, edge_feature, size=None):
        if torch.is_tensor(x):
            x = torch.matmul(x, self.weight1)
        else:
            x = (None if x[0] is None else torch.matmul(x[0], self.weight1),
                 None if x[1] is None else torch.matmul(x[1], self.weight1))

        edge_emb = torch.matmul(edge_feature, self.weight2)

        return self.propagate(edge_index, size=size, x=x, edge_emb=edge_emb)

    def message(self, x_j, edge_emb):
        x_j = torch.matmul(x_j, self.v)

        return edge_emb * x_j

    def update(self, aggr_out, x):
        if (isinstance(x, tuple) or isinstance(x, list)):
            x = x[1]

        aggr_out = torch.matmul(x, self.u) + aggr_out

        bn = nn.BatchNorm1d(aggr_out.shape[1]).to(x.device)
        aggr_out = bn(aggr_out)
        # aggr_out = self.batch_norm(aggr_out.view(-1, self.out_channels)).view(sz)
        # aggr_out = self.layer_norm(aggr_out)
        
        aggr_out = x + F.relu(aggr_out)
        
        return aggr_out


class GatedGCNNet(nn.Module):
    def __init__(self, in_channels, out_channels, edge_channels=1, spatial_channels=16, skip_connection=False):
        super(GatedGCNNet, self).__init__()
        self.skip_connection = skip_connection
        conv2_in_channels = spatial_channels * 2 if self.skip_connection else spatial_channels
        self.conv1 = GatedGCN(
            in_channels, spatial_channels, edge_channels=edge_channels, node_dim=1)
        self.conv2 = GatedGCN(
            conv2_in_channels, out_channels, edge_channels=edge_channels, node_dim=1)

    def forward(self, X, g):
        edge_index = g['edge_index']
        edge_weight = g['edge_weight']

        size = g['size']
        res_n_id = g['res_n_id']
        X = self.conv1(
            (X, X[:, res_n_id[0]]), edge_index[0], edge_feature=edge_weight[0], size=size[0])

        if self.skip_connection:
            X = torch.cat((F.leaky_relu(X), X), dim=-1)
        else:
            X = F.leaky_relu(X) 

        X = self.conv2(
            (X, X[:, res_n_id[1]]), edge_index[1], edge_feature=edge_weight[1], size=size[1])

        return X


class ClusterGatedGCNNet(nn.Module):
    def __init__(self, in_channels, out_channels, edge_channels=1, spatial_channels=16, skip_connection=False):
        super(ClusterGatedGCNNet, self).__init__()
        self.skip_connection = skip_connection
        conv2_in_channels = spatial_channels * 2 if self.skip_connection else spatial_channels
        self.conv1 = GatedGCN(
            in_channels, spatial_channels, edge_channels=edge_channels, node_dim=1)
        self.conv2 = GatedGCN(
            conv2_in_channels, out_channels, edge_channels=edge_channels, node_dim=1)

    def forward(self, X, g):
        edge_index = g['edge_index']
        edge_weight = g['edge_weight']
        X = self.conv1(X, edge_index, edge_feature=edge_weight)
        if self.skip_connection:
            X = torch.cat((F.leaky_relu(X), X), dim=-1)
        else:
            X = F.leaky_relu(X) 

        X = self.conv2(X, edge_index, edge_feature=edge_weight)

        return X


class MyEGNNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_channels, 
                 **kwargs):
        super(MyEGNNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_channels = edge_channels
        
        self.value = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.key = nn.Parameter(torch.Tensor(out_channels, out_channels))
        self.query = nn.Parameter(torch.Tensor(out_channels, out_channels))
        
        self.weight_e = nn.Parameter(torch.Tensor(edge_channels, out_channels))
        self.linear_att = nn.Linear(3 * out_channels, 1)
        self.linear_concat = nn.Linear(2 * out_channels, out_channels)

        # self.layer_norm = nn.LayerNorm(normalized_shape=out_channels)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.value)
        nn.init.xavier_uniform_(self.key)
        nn.init.xavier_uniform_(self.query)
        nn.init.xavier_uniform_(self.weight_e)


    def forward(self, x, edge_index, edge_feature, size=None):
        if torch.is_tensor(x):
            x = torch.matmul(x, self.value)
        else:
            x = (None if x[0] is None else torch.matmul(x[0], self.value),
                 None if x[1] is None else torch.matmul(x[1], self.value))

        edge_emb = torch.matmul(edge_feature, self.weight_e)

        return self.propagate(edge_index, size=size, x=x, edge_emb=edge_emb)

    def message(self, x_j, x_i, edge_emb):
        # cal att of shape [B, E, 1]
        query = torch.matmul(x_j, self.query)
        key = torch.matmul(x_i, self.key)

        att_feature = torch.cat([key, query, edge_emb.repeat(x_j.shape[0], 1, 1)], dim=-1)
        att = F.sigmoid(self.linear_att(att_feature))
        # gate of shape [1, E, C]
        gate = F.sigmoid(edge_emb)

        return att * x_j * gate

    def update(self, aggr_out, x):
        if (isinstance(x, tuple) or isinstance(x, list)):
            x = x[1]

        aggr_out = self.linear_concat(torch.cat([x, aggr_out], dim=-1))
        # aggr_out = self.layer_norm(aggr_out)
        # batch_norm = nn.BatchNorm1d(aggr_out.shape[1]).to(x.device)
        # aggr_out = batch_norm(aggr_out)
        
        return x + F.relu(aggr_out)


class MyEGNNNet(nn.Module):
    def __init__(self, in_channels, out_channels, edge_channels=1, spatial_channels=16, skip_connection=False):
        super(MyEGNNNet, self).__init__()
        self.skip_connection = skip_connection
        conv2_in_channels = spatial_channels * 2 if self.skip_connection else spatial_channels
        self.conv1 = MyEGNNConv(
            in_channels, spatial_channels, edge_channels=edge_channels, node_dim=1)
        self.conv2 = MyEGNNConv(
            conv2_in_channels, out_channels, edge_channels=edge_channels, node_dim=1)

    def forward(self, X, g):
        edge_index = g['edge_index']
        edge_weight = g['edge_weight']

        size = g['size']
        res_n_id = g['res_n_id']

        X = self.conv1(
            (X, X[:, res_n_id[0]]), edge_index[0], edge_feature=edge_weight[0], size=size[0])

        if self.skip_connection:
            X = torch.cat((F.leaky_relu(X), X), dim=-1)
        else:
            X = F.leaky_relu(X)

        X = self.conv2(
            (X, X[:, res_n_id[1]]), edge_index[1], edge_feature=edge_weight[1], size=size[1])

        return X

class ClusterMyEGNNNet(nn.Module):
    def __init__(self, in_channels, out_channels, edge_channels=1, spatial_channels=16, skip_connection=False):
        super(ClusterMyEGNNNet, self).__init__()
        self.skip_connection = skip_connection
        conv2_in_channels = spatial_channels * 2 if self.skip_connection else spatial_channels
        self.conv1 = MyEGNNConv(
            in_channels, spatial_channels, edge_channels=edge_channels, node_dim=1)
        self.conv2 = MyEGNNConv(
            conv2_in_channels, out_channels, edge_channels=edge_channels, node_dim=1)

    def forward(self, X, g):
        edge_index = g['edge_index']
        edge_weight = g['edge_weight']

        X = self.conv1(
            X, edge_index, edge_feature=edge_weight)

        if self.skip_connection:
            X = torch.cat((F.leaky_relu(X), X), dim=-1)
        else:
            X = F.leaky_relu(X)

        X = self.conv2(
            X, edge_index, edge_feature=edge_weight)

        return X