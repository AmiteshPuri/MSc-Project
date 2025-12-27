import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import APPNPConv, GraphConv, GATConv, SAGEConv
import utils
from torch.nn import functional as F, Parameter
import dgl.function as fn
from tqdm import tqdm
import numpy as np
import wandb
import copy
import pickle
import networkx as nx
import scipy.sparse as sp
import dgl
import os
from sklearn.metrics import f1_score
from torch_geometric.logging import log

global task
task = 'classification'


class ConfGNN(torch.nn.Module):
    def __init__(self, model, g, in_channels, hidden_channels, out_dim, activation, feat_drop, edge_drop, alpha, k):
        super(ConfGNN, self).__init__()
        self.model = model
        # num_classes = max(dataset.y).item() + 1
        # print(base_model)
        # self.confgnn = GNN(output_dim, hiddens, edge_drop, alpha, k)
        self.confgnn = APPNP(g, out_dim, hidden_channels, out_dim, activation, feat_drop, edge_drop, alpha, k)

    def forward(self, x, edge_index):
        with torch.no_grad():
            scores = self.model(x, edge_index)
        out = F.softmax(scores, dim=1)
        # if self.task == 'regression':
        #     out = scores
        # else:
        #     out = F.softmax(scores, dim = 1)
        adjust_scores = self.confgnn(out, edge_index)
        return adjust_scores, scores

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, activation, feat_drop, edge_drop, alpha, k):
        super().__init__()
        self.conv1 = GraphConv(in_channels, hidden_channels)
        self.propagate = APPNPConv(k, alpha, edge_drop)
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation

    def forward(self, features, edge_index):
        # prediction step
        h = features
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.conv1(h, edge_index).relu()
        # for layer in self.layers[1:-1]:
        #     h = self.activation(layer(h))
        # h = self.layers[-1](self.feat_drop(h))
        self.h = h
        # propagation step
        h = self.propagate(self.g, h)
        return h

class APPNP(nn.Module):
    def __init__(
            self,
            g,
            in_feats,
            hiddens,
            n_classes,
            activation,
            k,
            feat_drop,
            edge_drop,
            alpha,
    ):
        super(APPNP, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(nn.Linear(in_feats, hiddens[0]))
        # hidden layers
        for i in range(1, len(hiddens)):
            self.layers.append(nn.Linear(hiddens[i - 1], hiddens[i]))
        # output layer
        self.layers.append(nn.Linear(hiddens[-1], n_classes))
        self.activation = activation
        if feat_drop:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x
        self.propagate = APPNPConv(k, alpha, edge_drop)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, features, edge_index):
        # prediction step
        h = features
        h = self.feat_drop(h)
        # print('h_size', h.size())
        h = self.activation(self.layers[0](h))
        for layer in self.layers[1:-1]:
            h = self.activation(layer(h))
        h = self.layers[-1](self.feat_drop(h))
        self.h = h
        # propagation step
        h = self.propagate(self.g, h)
        return h

    def output(self, g, features):
        h = features
        for idx in range(len(self.layers) - 1):
            h = self.layers[idx](g, h).flatten(1)
        return self.layers[-1](g, h).mean(1)

    def shift_robust_output(self, idx_train, iid_train, alpha=1):
        return alpha * utils.cmd(self.h[idx_train, :], self.h[iid_train, :])
    
    def CMD_output(self, idx_train, iid_train, alpha=1):
        return alpha * utils.cmd(self.h[idx_train, :], self.h[iid_train, :])

    def MMD_output(self, idx_train, iid_train, alpha=1):
        return alpha * utils.MMD(self.h[idx_train, :], self.h[iid_train, :])

    def KLD_output(self, idx_train, iid_train, alpha=1):
        return alpha * utils.kld(self.h[idx_train, :], self.h[iid_train, :])
    
    def JSD_output(self, idx_train, iid_train, alpha=1):
        return alpha * utils.jsd(self.h[idx_train, :], self.h[iid_train, :])
    
    def EMD_output(self, idx_train, iid_train, alpha=1):
        return alpha * utils.emd(self.h[idx_train, :], self.h[iid_train, :])

class DAGNNConv(nn.Module):
    def __init__(self, in_dim, k):
        super(DAGNNConv, self).__init__()

        self.s = Parameter(torch.FloatTensor(in_dim, 1))
        self.k = k

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("sigmoid")
        nn.init.xavier_uniform_(self.s, gain=gain)

    def forward(self, graph, feats):
        with graph.local_scope():
            results = [feats]

            degs = graph.in_degrees().float()
            norm = torch.pow(degs, -0.5)
            norm = norm.to(feats.device).unsqueeze(1)

            for _ in range(self.k):
                feats = feats * norm
                graph.ndata["h"] = feats
                graph.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
                feats = graph.ndata["h"]
                feats = feats * norm
                results.append(feats)

            H = torch.stack(results, dim=1)
            S = torch.sigmoid(torch.matmul(H, self.s))
            S = S.permute(0, 2, 1)
            H = torch.matmul(S, H).squeeze()

            return H


class MLPLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, activation=None, dropout=0):
        super(MLPLayer, self).__init__()

        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        gain = 1.0
        if self.activation is F.relu:
            gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.linear.weight, gain=gain)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, feats):
        feats = self.dropout(feats)
        feats = self.linear(feats)
        if self.activation:
            feats = self.activation(feats)

        return feats


class DAGNN(nn.Module):
    def __init__(
        self,
        k,
        in_dim,
        hid_dim,
        out_dim,
        bias=True,
        activation=F.relu,
        dropout=0,
    ):
        super(DAGNN, self).__init__()
        self.mlp = nn.ModuleList()
        self.mlp.append(
            MLPLayer(
                in_dim=in_dim,
                out_dim=hid_dim,
                bias=bias,
                activation=activation,
                dropout=dropout,
            )
        )
        self.mlp.append(
            MLPLayer(
                in_dim=hid_dim,
                out_dim=out_dim,
                bias=bias,
                activation=None,
                dropout=dropout,
            )
        )
        self.dagnn = DAGNNConv(in_dim=out_dim, k=k)

    def forward(self, graph, feats):
        h = feats
        for layer in self.mlp:
            h = layer(h)
        h = self.dagnn(graph, h)
        self.h = h
        return h

    def output(self, g, features):
        h = features
        for idx in range(len(self.layers)-1):
            h = self.layers[idx](g, h).flatten(1)
        return self.layers[-1](g, h).mean(1)


    def shift_robust_output(self, idx_train, iid_train, alpha=1):
        return alpha * utils.cmd(self.h[idx_train, :], self.h[iid_train, :])
    
    def CMD_output(self, idx_train, iid_train, alpha=1):
        return alpha * utils.cmd(self.h[idx_train, :], self.h[iid_train, :])

    def MMD_output(self, idx_train, iid_train, alpha=1):
        return alpha * utils.MMD(self.h[idx_train, :], self.h[iid_train, :])

    def KLD_output(self, idx_train, iid_train, alpha=1):
        return alpha * utils.kld(self.h[idx_train, :], self.h[iid_train, :])
    
    def JSD_output(self, idx_train, iid_train, alpha=1):
        return alpha * utils.jsd(self.h[idx_train, :], self.h[iid_train, :])
    
    def EMD_output(self, idx_train, iid_train, alpha=1):
        return alpha * utils.emd(self.h[idx_train, :], self.h[iid_train, :])
    

class GCN_(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN_, self).__init__()
        self.layers = nn.ModuleList()
        self.g = g
        # print(in_feats, n_hidden, n_classes)
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=None))

        # hidden layers
        self.activation = activation
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=None))
        # output layer hidden units -> n_classes
        self.layers.append(GraphConv(n_hidden, n_classes, activation=None))  # activation None
        self.fcs = nn.ModuleList([nn.Linear(n_hidden, n_hidden, bias=True), nn.Linear(n_hidden, 2, bias=True)])
        self.disc = GraphConv(n_hidden, 2, activation=None)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features, edge_index):
        h = features
        for idx, layer in enumerate(self.layers[:-1]):
            h = layer(self.g, h)
            h = self.activation(h)
            h = self.dropout(h)
        self.h = h

        return self.layers[-1](self.g, h)

    def shift_robust_output(self, idx_train, iid_train, alpha=1):
        return alpha * utils.cmd(self.h[idx_train, :], self.h[iid_train, :])
    
    def CMD_output(self, idx_train, iid_train, alpha=1):
        return alpha * utils.cmd(self.h[idx_train, :], self.h[iid_train, :])

    def MMD_output(self, idx_train, iid_train, alpha=1):
        return alpha * utils.MMD(self.h[idx_train, :], self.h[iid_train, :])
    
    def KLD_output(self, idx_train, iid_train, alpha=1):
        return alpha * utils.kld(self.h[idx_train, :], self.h[iid_train, :])
    
    def JSD_output(self, idx_train, iid_train, alpha=1):
        return alpha * utils.jsd(self.h[idx_train, :], self.h[iid_train, :])
    
    def EMD_output(self, idx_train, iid_train, alpha=1):
        return alpha * utils.emd(self.h[idx_train, :], self.h[iid_train, :])

    def output(self, features):
        h = features
        for layer in self.layers[:-1]:
            h = layer(self.g, h)
        return h


class GAT_(nn.Module):
    def __init__(self, in_size, hid_size, out_size, heads):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        # two-layer GAT
        self.gat_layers.append(
            GATConv(
                in_size,
                hid_size,
                heads[0],
                feat_drop=0.6,
                attn_drop=0.6,
                activation=F.elu,
            )
        )
        self.gat_layers.append(
            GATConv(
                hid_size * heads[0],
                out_size,
                heads[1],
                feat_drop=0.6,
                attn_drop=0.6,
                activation=None,
            )
        )

    def forward(self, g, inputs):
        h = inputs
        for i, layer in enumerate(self.gat_layers):
            h = layer(g, h)
            if i == len(self.gat_layers) - 1:  # last layer
                h = h.mean(1)
            else:  # other layer(s)
                h = h.flatten(1)
        self.h = h
        return h

    def output(self, g, features):
        h = features
        for idx in range(len(self.layers)-1):
            h = self.layers[idx](g, h).flatten(1)
        return self.layers[-1](g, h).mean(1)

    def shift_robust_output(self, idx_train, iid_train, alpha=1):
        return alpha * utils.cmd(self.h[idx_train, :], self.h[iid_train, :])
    
    def CMD_output(self, idx_train, iid_train, alpha=1):
        return alpha * utils.cmd(self.h[idx_train, :], self.h[iid_train, :])

    def MMD_output(self, idx_train, iid_train, alpha=1):
        return alpha * utils.MMD(self.h[idx_train, :], self.h[iid_train, :])
    
    def KLD_output(self, idx_train, iid_train, alpha=1):
        return alpha * utils.kld(self.h[idx_train, :], self.h[iid_train, :])
    
    def JSD_output(self, idx_train, iid_train, alpha=1):
        return alpha * utils.jsd(self.h[idx_train, :], self.h[iid_train, :])
    
    def EMD_output(self, idx_train, iid_train, alpha=1):
        return alpha * utils.emd(self.h[idx_train, :], self.h[iid_train, :])




class GraphSAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GraphSAGE-mean
        self.layers.append(SAGEConv(in_size, hid_size, "gcn"))
        self.layers.append(SAGEConv(hid_size, out_size, "gcn"))
        self.dropout = nn.Dropout(0.5)

    def forward(self, graph, x):
        h = self.dropout(x)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        self.h= h
        return h

    def output(self, features):
        h = features
        for layer in self.layers[:-1]:
            h = layer(self.g, h)
        return h

    def shift_robust_output(self, idx_train, iid_train, alpha=1):
        return alpha * utils.cmd(self.h[idx_train, :], self.h[iid_train, :])
    
    def CMD_output(self, idx_train, iid_train, alpha=1):
        return alpha * utils.cmd(self.h[idx_train, :], self.h[iid_train, :])

    def MMD_output(self, idx_train, iid_train, alpha=1):
        return alpha * utils.MMD(self.h[idx_train, :], self.h[iid_train, :])

    def KLD_output(self, idx_train, iid_train, alpha=1):
        return alpha * utils.kld(self.h[idx_train, :], self.h[iid_train, :])
    
    def JSD_output(self, idx_train, iid_train, alpha=1):
        return alpha * utils.jsd(self.h[idx_train, :], self.h[iid_train, :])
    
    def EMD_output(self, idx_train, iid_train, alpha=1):
        return alpha * utils.emd(self.h[idx_train, :], self.h[iid_train, :])


