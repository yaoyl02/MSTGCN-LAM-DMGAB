# For relative import
import os
import sys

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJ_DIR)
print(PROJ_DIR)
import argparse

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
import wandb

from models.MSTGCN import MSTGCN_submodule
from models.ASTGCN import ASTGCN
from models.STGMN import STGMN
from models.fusiongraph import FusionGraphModel
from datasets.air import *
from util import *

from pytorch_lightning.utilities.cloud_io import load

parser = argparse.ArgumentParser()  # 创建解析器
args = parser.parse_args()  # 解析参数

# gpu_num = 0                                                 # set the GPU number of your server.
gpu_num = None
os.environ['WANDB_MODE'] = 'online'  # select one from ['online','offline']

hyperparameter_defaults = dict(
    server=dict(
        gpu_id=0,
    ),
    graph=dict(
        use=['dist', 'neighb', 'distri', 'tempp', 'func'],
        # select no more than five graphs from ['dist', 'neighb', 'distri', 'tempp', 'func'].
        # use=['dist'],
        fix_weight=False,  # if True, the weight of each graph is fixed.
        tempp_diag_zero=True,  # if Ture, the values of temporal pattern similarity weight matrix turn to 0.
        matrix_weight=True,  # if True, turn the weight matrices trainable.
        distri_type='exp',
        # select one from ['kl', 'ws', 'exp']: 'kl' is for Kullback-Leibler divergence, 'ws' is for Wasserstein, and 'exp' is for expotential fitting
        func_type='ours',
        # select one from ['ours', 'others'], 'others' is for the functionality graph proposed by "Spatiotemporal Multi-Graph Convolution Network for Ride-hailing Demand Forecasting"
        attention=True,  # if True, the SG-ATT is used.
    ),
    model=dict(
        # TODO: check batch_size
        use='STGMN'
    ),
    data=dict(
        in_dim=1,
        out_dim=1,
        hist_len=24,
        # hist_len=12,
        pred_len=12,
        type='metr-la',
        augment=True,

    ),
    STMGCN=dict(
        use_fusion_graph=True,
    ),
    train=dict(
        seed=35,
        epoch=1,
        batch_size=32,
        lr=1e-4,
        weight_decay=1e-4,
        M=12,
        d=12,
        bn_decay=0.1,
    )
)

wandb_proj = 'metr-la'  # mlstgcn-master
wandb.init(config=hyperparameter_defaults, project=wandb_proj, settings=wandb.Settings(start_method='thread'))
wandb_logger = WandbLogger()
config = wandb.config

pl.utilities.seed.seed_everything(config['train']['seed'])

gpu_id = config['server']['gpu_id']
# device = 'cuda:%d' % gpu_id
device = torch.device('cuda:%d' % gpu_id if torch.cuda.is_available() else 'cpu')

if config['data']['type'] == 'metr-la':
    root_dir = 'data'
    metr_la_data_dir = os.path.join(root_dir, 'temporal_data/metr-la-12-with_hist')
    metr_la_graph_dir = os.path.join(root_dir, 'graph/metr-la')
    metr_la_aug_data_dir = os.path.join(root_dir, 'temporal_data/metr-la-12-with_hist')
else:
    raise NotImplementedError

if config['data']['type'] == 'metr-la':
    graph = AirGraph(metr_la_graph_dir, config['graph'], gpu_id)
    train_set = Air(metr_la_data_dir, 'train',config['data']['augment'])
    val_set = Air(metr_la_data_dir, 'val',config['data']['augment'])
    test_set = Air(metr_la_data_dir, 'test',config['data']['augment'])
else:
    raise NotImplementedError

scaler = train_set.scaler


# 说明数据是经过归一化之后再进行后续处理

class LightningData(LightningDataModule):
    def __init__(self, train_set, val_set, test_set):
        super().__init__()
        self.batch_size = config['train']['batch_size']
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set

    # 加载数据类：DataLoader
    # [数据加载器。组合数据集和采样器，并在数据集上提供单进程或多进程迭代器。]
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=0,
                          pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=0,
                          pin_memory=True, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=0,
                          pin_memory=True, drop_last=True)


class LightningModel(LightningModule):
    def __init__(self, scaler, fusiongraph):
        super().__init__()

        self.scaler = scaler
        # self.grapg=graph_final
        self.fusiongraph = fusiongraph

        self.metric_lightning = LightningMetric()  # 评价指标

        self.loss = nn.L1Loss(reduction='mean')

        if config['model']['use'] == 'ASTGCN':
            # self.model = ASTGCN_submodule(gpu_id, fusiongraph, config['data']['in_dim'], config['data']['hist_len'], config['data']['pred_len'])
            self.model = ASTGCN(gpu_id, fusiongraph, config['data']['in_dim'], config['data']['hist_len'],
                                config['data']['pred_len'])

            for p in self.model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
                else:
                    nn.init.uniform_(p)
        elif config['model']['use'] == 'MSTGCN':
            self.model = MSTGCN_submodule(gpu_id, fusiongraph, config['data']['in_dim'], config['data']['hist_len'],
                                          config['data']['pred_len'])
            for p in self.model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
                else:
                    nn.init.uniform_(p)
        elif config['model']['use'] == 'STGMN':
            self.model = STGMN(gpu_id, config['data']['in_dim'], config['data']['hist_len'], config['data']['pred_len'],
                               fusiongraph)
            for p in self.model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
                else:
                    nn.init.uniform_(p)
        else:
            raise NotImplementedError

        self.log_dict(config)

    def forward(self, x):
        return self.model(x)

    def _run_model(self, batch):
        x, y = batch

        y_hat = self(x)

        y_hat = self.scaler.inverse_transform(y_hat)

        loss = masked_mae(y_hat, y, 0.0)

        return y_hat, y, loss

    def training_step(self, batch, batch_idx):
        y_hat, y, loss = self._run_model(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, y, loss = self._run_model(batch)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        y_hat, y, loss = self._run_model(batch)
        self.metric_lightning(y_hat.cpu(), y.cpu())
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_epoch_end(self, outputs):
        test_metric_dict = self.metric_lightning.compute()
        self.log_dict(test_metric_dict)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=config['train']['lr'], weight_decay=config['train']['weight_decay'])


def main():
    fusiongraph = FusionGraphModel(graph, gpu_id, config['graph'], config['data'], config['train']['M'],
                                   config['train']['d'], config['train']['bn_decay'])

    lightning_data = LightningData(train_set, val_set, test_set)

    lightning_model = LightningModel(scaler, fusiongraph)

    trainer = Trainer(
        logger=wandb_logger,
        # gpus=[gpu_id],
        gpus=None,
        max_epochs=config['train']['epoch'],
        # TODO
        # precision=16,
    )

    trainer.fit(lightning_model, lightning_data)
    trainer.test(lightning_model, datamodule=lightning_data)
    print('Graph USE', config['graph']['use'])
    print('Data', config['data'])


if __name__ == '__main__':
    main()
