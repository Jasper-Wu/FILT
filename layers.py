import torch
import torch.nn as nn

from torch_geometric.nn.inits import uniform
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax

class TemporalGraphEncoder(MessagePassing):
    def __init__(self, in_channels, out_channels, num_relations, root_weight=True, bias=True, args=None, **kwargs):
        super(TemporalGraphEncoder, self).__init__(aggr='add', **kwargs)

        self.args = args

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations

        self.proj = nn.Linear(in_channels, out_channels)
        self.leakyrelu = nn.LeakyReLU()

        if root_weight:
            self.root = nn.Parameter(torch.Tensor(out_channels, out_channels))
        else:
            self.register_parameter('root', None)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.root is not None:
            nn.init.xavier_uniform_(self.root)
        if self.bias is not None:
            nn.init.xavier_uniform_(self.bias)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x, edge_index, edge_type, rel_emb, ts_emb, time_diff, edge_norm=None, size=None):
        self.rel_emb = rel_emb
        return self.propagate(edge_index, x=x, edge_type=edge_type, ts_emb=ts_emb, time_diff=time_diff,
                            size=size, edge_norm=edge_norm)

    def message(self, x_j, edge_type, ts_emb, time_diff, edge_index_i, size_i):
        '''
        params:
            x_j: [E, D]
            self.rel_emb: [E, D]
            time_diff: [E, Q]
            edge_index_i: [E]
        return:
            msg: [Q, E, D]
        '''
        x_comb = torch.cat([x_j, self.rel_emb], dim=-1)
        out = self.proj(x_comb)
        alpha = softmax(src=time_diff, index=edge_index_i, num_nodes=size_i).t()
        msg = out.unsqueeze(0) * alpha.unsqueeze(-1)
        assert msg.shape == (time_diff.shape[1], len(x_j), self.out_channels)
        return msg

    def update(self, aggr_out, x):
        if self.root is not None:
            if x is None:
                out = aggr_out + self.root
            else:
                out = aggr_out + torch.matmul(x, self.root)
        if self.bias is not None:
            out = out + self.bias
        if (self.root is None) and (self.bias is None):
            return aggr_out
        return out

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_relations)


    