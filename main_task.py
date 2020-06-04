import os
import time
import argparse
import json
import math
import torch
import torch_geometric
from torch.utils.data import DataLoader, TensorDataset, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from argparse import Namespace
from torch_geometric.data import Data, Batch, NeighborSampler, ClusterData, ClusterLoader
import torch.nn as nn
import torch.distributed as dist
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tgcn import TGCN
from stgcn import STGCN
from sandwish import Sandwich
from preprocess import generate_dataset, load_nyc_sharing_bike_data, load_metr_la_data, get_normalized_adj, load_sf_sample_data
from dataset import NeighborSampleDataset,ClusterDataset
from base_task import add_config_to_argparse, BaseConfig, BasePytorchTask, \
    LOSS_KEY, BAR_KEY, SCALAR_LOG_KEY, VAL_SCORE_KEY



class STConfig(BaseConfig):
    def __init__(self):
        super(STConfig, self).__init__()
        # 1. Reset base config variables:
        self.max_epochs = 1000
        self.early_stop_epochs = 30

        # 2. set spatial-temporal config variables:
        self.model = 'tgcn'  # choices: tgcn, stgcn, gwnet
        self.dataset = 'metr'  # choices: metr, nyc
        self.data_dir = './data/METR-LA'  # choices: ./data/METR-LA, ./data/NYC-Sharing-Bike
        self.gcn = 'sage'  # choices: sage, gat
        self.gcn_package = 'ours'  # choices: pyg, ours
        self.gcn_partition = 'none'  # choices: none, cluster, sample
        self.batch_size = 32  # per-gpu training batch size, real_batch_size = batch_size * num_gpus * grad_accum_steps
        self.loss = 'mse'  # choices: mse, mae
        self.num_timesteps_input = 12  # the length of the input time-series sequence
        self.num_timesteps_output = 3  # the length of the output time-series sequence
        self.skip_connection = False  # whether to use skip connection in gcn 
        self.lr = 1e-3  # the learning rate
        self.hidden_size = 64 # the hidden size
        self.use_gcn = True # whether to use gcn in sandwish or pure k-rnn


def get_model_class(model):
    return {
        'tgcn': TGCN,
        'sandwish': Sandwich,
        # 'stgcn': STGCN,
        # 'gwnet': GWNET,
    }.get(model)


def get_loss_func(loss):
    return {
        'mse': nn.MSELoss(),
        'mae': nn.L1Loss(),
    }.get(loss)



class WrapperNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        model_class = get_model_class(config.model)
        gcn_partition = None if config.gcn_partition is 'none' else config.gcn_partition
        self.net = model_class(
            num_nodes = config.num_nodes,
            num_edges = config.num_edges,
            num_features = config.num_features,
            num_edge_features = config.num_edge_features,
            num_timesteps_input = config.num_timesteps_input,
            num_timesteps_output = config.num_timesteps_output,
            skip_connection = config.skip_connection,
            gcn_type = config.gcn,
            gcn_partition = gcn_partition,
            hidden_size = config.hidden_size,
            use_gcn = config.use_gcn
        )

        self.register_buffer('edge_index', torch.LongTensor(
            2, config.num_edges))
        if config.num_edge_features == 1:
            self.register_buffer('edge_weight', torch.FloatTensor(
                config.num_edges))
        else:
            self.register_buffer('edge_weight', torch.FloatTensor(
                config.num_edges, config.num_edge_features))            

    def init_graph(self, edge_index, edge_weight):
        self.edge_index.copy_(edge_index)
        self.edge_weight.copy_(edge_weight)

    def forward(self, X, g):
        return self.net(X, g)


class SpatialTemporalTask(BasePytorchTask):
    def __init__(self, config):
        super(SpatialTemporalTask, self).__init__(config)
        self.log('Intialize {}'.format(self.__class__))

        self.init_data()
        self.loss_func = get_loss_func(config.loss)

        self.log('Config:\n{}'.format(
            json.dumps(self.config.to_dict(), ensure_ascii=False, indent=4)
        ))


    def init_data(self, data_dir=None):
        if data_dir is None:
            data_dir = self.config.data_dir

        if self.config.dataset == "metr":
            A, X, means, stds = load_metr_la_data(data_dir)
        elif self.config.dataset == 'nyc-bike':
            A, X, means, stds = load_nyc_sharing_bike_data(data_dir)
        elif self.config.dataset == 'sf-example':
            edge_index, edge_weight, X, shenzhenmask = load_sf_sample_data(data_dir)
            self.node_mask = shenzhenmask

        X = X.astype(np.float32)
        split_line1 = int(X.shape[2] * 0.2)
        split_line2 = int(X.shape[2] * 0.4)
        train_original_data = X[:, :, :split_line1]
        val_original_data = X[:, :, split_line1:split_line2]
        test_original_data = X[:, :, split_line2:]

        self.training_input, self.training_target = generate_dataset(train_original_data,
            num_timesteps_input=self.config.num_timesteps_input, num_timesteps_output=self.config.num_timesteps_output
        )
        self.val_input, self.val_target = generate_dataset(val_original_data,
            num_timesteps_input=self.config.num_timesteps_input, num_timesteps_output=self.config.num_timesteps_output
        )
        self.test_input, self.test_target = generate_dataset(test_original_data,
            num_timesteps_input=self.config.num_timesteps_input, num_timesteps_output=self.config.num_timesteps_output
        )

        if self.config.dataset in ['metr','nyc-bike']:
            self.A = torch.from_numpy(A)
            self.sparse_A = self.A.to_sparse()
            self.edge_index = self.sparse_A._indices()
            self.edge_weight = self.sparse_A._values()

            # set config attributes for model initialization
            self.config.num_nodes = self.A.shape[0]
            self.config.num_edges = self.edge_weight.shape[0]
            self.config.num_edge_features = 1
            self.config.num_features = self.training_input.shape[3]
            self.log('Total nodes: {}'.format(self.config.num_nodes))
            self.log('Average degree: {:.3f}'.format(self.config.num_edges / self.config.num_nodes))

            contains_self_loops = torch_geometric.utils.contains_self_loops(self.edge_index)
            self.log('Contains self loops: {}, but we add them.'.format(contains_self_loops))
            if not contains_self_loops:
                self.edge_index, self.edge_weight = torch_geometric.utils.add_self_loops(
                    self.edge_index, self.edge_weight,
                    num_nodes=self.config.num_nodes
                )
        
        elif self.config.dataset in ['sf-example']:
            self.edge_index = torch.from_numpy(edge_index)
            self.edge_weight = torch.from_numpy(edge_weight)

            # set config attributes for model initialization
            self.config.num_nodes = len(shenzhenmask)
            self.config.num_edges = self.edge_weight.shape[0]
            self.config.num_edge_features = self.edge_weight.shape[1]   
            self.config.num_features = self.training_input.shape[3] 
            self.log('Total nodes: {}'.format(self.config.num_nodes))
            self.log('Average degree: {:.3f}'.format(self.config.num_edges / self.config.num_nodes))

            # contains_self_loops = torch_geometric.utils.contains_self_loops(self.edge_index)
            # self.log('Contains self loops: {}, but we add them.'.format(contains_self_loops))
            # if not contains_self_loops:
            #     self.edge_index, _ = torch_geometric.utils.add_self_loops(
            #         self.edge_index,
            #         num_nodes=self.config.num_nodes
            #     )


    def make_sample_dataloader(self, X, y, shuffle=False, use_dist_sampler=False):
        # return a data loader based on neighbor sampling
        # TODO: impl. distributed dataloader
        assert self.config.gcn_partition in ["sample","cluster"]
        Sampler = {"sample": NeighborSampleDataset,
                    "cluster": ClusterDataset}.get(self.config.gcn_partition)
            
        dataset = Sampler(
            X, y, self.edge_index, self.edge_weight, self.config.num_nodes, self.config.batch_size,
            shuffle=shuffle, use_dist_sampler=use_dist_sampler
        )

        return DataLoader(dataset, batch_size=None)

    def build_train_dataloader(self):
        return self.make_sample_dataloader(
            self.training_input, self.training_target, shuffle=True, use_dist_sampler=True
        )

    def build_val_dataloader(self):
        return self.make_sample_dataloader(self.val_input, self.val_target)

    def build_test_dataloader(self):
        return self.make_sample_dataloader(self.test_input, self.test_target)

    def build_optimizer(self, model):
        return torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

    def train_step(self, batch, batch_idx):
        X, y, g, rows = batch
        y_hat = self.model(X, g)   # n_rows, cent_id, output_dim
        # n_rows,n_cent_id,output_dim = y_hat.size()
        assert(y.size() == y_hat.size())
        # if self.config.dataset in ['sf-example']:
        #    mask_rows = self.node_mask[g['cent_n_id']]
        #    mask_rows = mask_rows.reshape(1,-1,1).expand(n_rows,n_cent_id,output_dim)
        #    y_hat = torch.masked_select(y_hat, mask_rows) 
        #    y = torch.masked_select(y, mask_rows)
        #    if y_hat.size()[0] == 0:
        loss = self.loss_func(y_hat, y)
        loss_i = loss.item()  # scalar loss

        # # debug distributed sampler
        # if batch_idx == 0:
        #     self.log('indices: {}'.format(rows))
        #     self.log('g.cent_n_id: {}'.format(g['cent_n_id']))

        return {
            LOSS_KEY: loss,
            BAR_KEY: { 'train_loss': loss_i },
            SCALAR_LOG_KEY: { 'train_loss': loss_i }
        }

    def eval_step(self, batch, batch_idx, tag):
        X, y, g, rows = batch

        y_hat = self.model(X, g)
        assert(y.size() == y_hat.size())

        out_dim = y.size(-1)

        index_ptr = torch.cartesian_prod(
            torch.arange(rows.size(0)),
            torch.arange(g['cent_n_id'].size(0)),
            torch.arange(out_dim)
        )

        label = pd.DataFrame({
            'row_idx': rows[index_ptr[:, 0]].data.cpu().numpy(),
            'node_idx': g['cent_n_id'][index_ptr[:, 1]].data.cpu().numpy(),
            'feat_idx': index_ptr[:, 2].data.cpu().numpy(),
            'val': y[index_ptr.t().chunk(3)].squeeze(dim=0).data.cpu().numpy()
        })

        pred = pd.DataFrame({
            'row_idx': rows[index_ptr[:, 0]].data.cpu().numpy(),
            'node_idx': g['cent_n_id'][index_ptr[:, 1]].data.cpu().numpy(),
            'feat_idx': index_ptr[:, 2].data.cpu().numpy(),
            'val': y_hat[index_ptr.t().chunk(3)].squeeze(dim=0).data.cpu().numpy()
        })

        pred = pred.groupby(['row_idx', 'node_idx', 'feat_idx']).mean()
        label = label.groupby(['row_idx', 'node_idx', 'feat_idx']).mean()

        return {
            'label': label,
            'pred': pred,
        }

    def eval_epoch_end(self, outputs, tag):
        pred = pd.concat([x['pred'] for x in outputs], axis=0)
        label = pd.concat([x['label'] for x in outputs], axis=0)

        pred = pred.groupby(['row_idx', 'node_idx', 'feat_idx']).mean().values
        label = label.groupby(['row_idx', 'node_idx', 'feat_idx']).mean().values

        loss = np.mean(pred - label) ** 2

        pred = pred[label>0]
        label = label[label>0]

        mape = np.mean(np.abs(np.expm1(pred) - np.expm1(label)) / np.expm1(label))

        out = {
            BAR_KEY: { '{}_loss'.format(tag) : loss },
            SCALAR_LOG_KEY: { '{}_loss'.format(tag) : loss ,
                            '{}_mape'.format(tag) : mape},
            VAL_SCORE_KEY: -loss,  # a larger score corresponds to a better model
        }

        return out

    def val_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'val')

    def val_epoch_end(self, outputs):
        return self.eval_epoch_end(outputs, 'val')

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'test')

    def test_epoch_end(self, outputs):
        return self.eval_epoch_end(outputs, 'test')


if __name__ == '__main__':
    start_time = time.time()

    # build argument parser and config
    config = STConfig()
    parser = argparse.ArgumentParser(description='Spatial-Temporal-Task')
    add_config_to_argparse(config, parser)

    # parse arguments to config
    args = parser.parse_args()
    config.update_by_dict(args.__dict__)

    # build task
    task = SpatialTemporalTask(config)

    # Set random seed before the initialization of network parameters
    # Necessary for distributed training
    task.set_random_seed()
    net = WrapperNet(task.config)
    net.init_graph(task.edge_index, task.edge_weight)

    if task.config.skip_train:
        task.init_model_and_optimizer(net)
    else:
        task.fit(net)

    # Resume the best checkpoint for evaluation
    task.resume_best_checkpoint()
    val_eval_out = task.val_eval()
    test_eval_out = task.test_eval()
    task.log('Best checkpoint (epoch={}, {}, {})'.format(
        task._passed_epoch, val_eval_out[BAR_KEY], test_eval_out[BAR_KEY]))

    task.log('Training time {}s'.format(time.time() - start_time))