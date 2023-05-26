import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from layers import TemporalGraphEncoder
from utils import complex_score

class Induc(nn.Module):

    def __init__(self, entity_embedding_dim, relation_embedding_dim, time_embedding_dim, 
                num_entities, num_relations, num_ts, args, entity_embedding, relation_embedding, time_embedding, mode):

        super(Induc, self).__init__()

        self.args = args

        self.entity_embedding_dim = entity_embedding_dim
        self.relation_embedding_dim = relation_embedding_dim
        self.time_embedding_dim = time_embedding_dim
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.num_times = num_ts

        self.entity_embedding = nn.Embedding(num_entities, entity_embedding_dim)
        if not self.args.rev_rel_emb:
            self.relation_embedding = nn.Parameter(torch.Tensor(num_relations, relation_embedding_dim))
        elif self.args.rev_rel_emb:
            self.relation_embedding = nn.Parameter(torch.Tensor(2 * num_relations, relation_embedding_dim))

        nn.init.xavier_uniform_(self.relation_embedding, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.entity_embedding.weight, gain=nn.init.calculate_gain('relu'))

        if self.args.pre_train:
            self.entity_embedding.weight.data.copy_(entity_embedding.clone().detach())
            self.relation_embedding.data.copy_(relation_embedding.clone().detach())
            if time_embedding is not None:
                self.time_embedding.data.copy_(time_embedding.clone().detach())
            assert self.relation_embedding.shape == relation_embedding.shape

            if not self.args.fine_tune:
                self.entity_embedding.weight.requires_grad = False
                self.relation_embedding.requires_grad = False

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)

        if mode == 'tw':
            self.gnn = TemporalGraphEncoder(self.entity_embedding_dim + self.relation_embedding_dim, self.entity_embedding_dim,
                                self.num_relations * 2, root_weight=False, bias=False)

        self.score_function = self.args.score_function
        self.time_granularity = 24

        if self.args.res_cof:
            self.res_cof_out = nn.Parameter(torch.Tensor([args.res_cof]))
        else:
            self.res_cof_out = 1.0

        self.concept_model = ConceptModel(in_dim=self.entity_embedding_dim, hidden_size=self.entity_embedding_dim, res_cof=self.args.res_cof)

    def graph_preprocess(self, quads, use_cuda, diff_reverse=False):
        """
        edge_index order: [[src, dst], 
                           [dst, src]]
        return:
            uniq_v: entity index
            node_id: for entity embedding, [num_entity]
            rel_index: for relation embedding, [E]
            edge_index: graph edges, double of num_quads
            edge_type:
        """
        # Pre-process
        if quads.shape[1] == 3:
            src, rel, dst = quads.transpose()
        else:
            src, rel, dst, t = quads.transpose()
        uniq_v, edges = np.unique((src, dst), return_inverse=True)
        src, dst = np.reshape(edges, (2, -1))

        if not diff_reverse:
            rel_index = np.concatenate((rel, rel))
        else:
            rel_index = np.concatenate([rel, rel + self.num_relations])

        src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
        rel = np.concatenate((rel, rel + self.num_relations))

        # Torch
        node_id = torch.LongTensor(uniq_v)
        edge_index = torch.stack((
            torch.LongTensor(src),
            torch.LongTensor(dst)
        ))
        edge_type = torch.LongTensor(rel)

        if use_cuda:
            node_id = node_id.cuda()
            edge_index = edge_index.cuda()
            edge_type = edge_type.cuda()
        return uniq_v, node_id, rel_index, edge_index, edge_type

    def forward_tw(self, unseen_entity, quads, query_t, use_cuda, query_r=None, cover_ent_emb=None):
        '''
        return:
            unseen_entity_embedding: [num_query_t, dim]
        '''
        uniq_v, node_id, rel_index, edge_index, edge_type = self.graph_preprocess(quads, use_cuda, diff_reverse=self.args.rev_rel_emb)
        
        unseen_index = np.where(uniq_v == unseen_entity)[0][0]

        quad_t = quads[:, 3]
        t_diff = (quad_t[:, np.newaxis] - query_t) / self.time_granularity
        t_diff = np.tile(t_diff, (2, 1))
        t_diff_abs = np.absolute(t_diff)
        t_diff_abs[t_diff_abs == 0] = 0.2
        t_diff_inv = 1. / t_diff_abs
        t_diff_inv = torch.Tensor(t_diff_inv)
        if use_cuda:
            t_diff_inv = t_diff_inv.cuda()

        if cover_ent_emb is None:
            x = self.entity_embedding(node_id)
        else:
            x = cover_ent_emb[1][node_id]
        rel_emb = self.relation_embedding[rel_index]

        unseen_entity_embedding = self.gnn(x, edge_index, edge_type, rel_emb, None, t_diff_inv)[:, unseen_index, :]  # [Q, num_node, D]

        if cover_ent_emb is not None:
            unseen_entity_embedding = self.dropout(unseen_entity_embedding + self.res_cof_out * cover_ent_emb[0][unseen_entity])
        else:
            unseen_entity_embedding = self.dropout(unseen_entity_embedding) # [Q, D]

        return unseen_entity_embedding, self.res_cof_out

    def score_loss_td_batch(self, unseen_entity, unseen_entity_embedding_td, quads, use_cuda, target=None):
        num_query_t = len(unseen_entity_embedding_td)
        assert num_query_t == len(quads) // (1 + self.args.negative_sample)

        head_embeddings = self.entity_embedding(quads[:, 0])
        relation_embeddings = self.relation_embedding[quads[:, 1]]
        tail_embeddings = self.entity_embedding(quads[:, 2])
        if self.args.rev_rel_emb:
            relation_embeddings_inv = self.relation_embedding[quads[:, 1] + self.num_relations]

        head_mask = quads[:, 0] == unseen_entity
        tail_mask = quads[:, 2] == unseen_entity
        assert head_mask.sum() + tail_mask.sum() == len(quads)

        head_mask_first = head_mask[:num_query_t]
        tail_mask_first = tail_mask[:num_query_t]
        head_embeddings[head_mask] = unseen_entity_embedding_td[head_mask_first].repeat(1 + self.args.negative_sample, 1)
        tail_embeddings[tail_mask] = unseen_entity_embedding_td[tail_mask_first].repeat(1 + self.args.negative_sample, 1)

        len_positive_triplets = int(len(quads) / (self.args.negative_sample + 1))

        if self.score_function == 'ComplEx':
            dim = unseen_entity_embedding_td.shape[1]
            score = complex_score(head_embeddings, relation_embeddings, tail_embeddings)
            positive_score = score[:len_positive_triplets]
            negative_score = score[len_positive_triplets:]
            if self.args.rev_rel_emb:
                score_inv = complex_score(tail_embeddings, relation_embeddings_inv, head_embeddings)
                negative_score = torch.cat([negative_score, score_inv[len_positive_triplets:]])

        if not self.args.rev_rel_emb:
            y = torch.ones(len_positive_triplets * self.args.negative_sample)
            positive_score = positive_score.repeat(self.args.negative_sample)
        else:
            y = torch.ones(2 * len_positive_triplets * self.args.negative_sample)
            positive_score = torch.cat([positive_score.repeat(self.args.negative_sample), 
                                        score_inv[:len_positive_triplets].repeat(self.args.negative_sample)])

        if use_cuda:
            y = y.cuda()

        loss = F.margin_ranking_loss(positive_score, negative_score, y, margin=self.args.margin)

        return loss

class ConceptModel(nn.Module):
    def __init__(self, in_dim=100, hidden_size=64, res_cof=0):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_size = hidden_size

        self.fc_ent = nn.Linear(in_dim, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_ent.weight)

        self.fc_skip_out = nn.Linear(in_dim, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_skip_out.weight)

        self.fc_skip_in = nn.Linear(in_dim, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_skip_in.weight)

        self.fc_out_gnn = nn.Linear(in_dim, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_out_gnn.weight)

        self.leaky_relu = nn.LeakyReLU()
        self.softmax_along_0 = torch.nn.Softmax(dim=0)
        self.softmax_along_1 = torch.nn.Softmax(dim=1)

        self.res_cof = res_cof
        if res_cof:
            self.res_cof_in = nn.Parameter(torch.Tensor([res_cof]))
        else:
            self.res_cof_in = 0

    def forward(self, x, ent2sec_mat, mask_index=None):
        '''
        params:
            x: 
            ent2sec_mat: [E, S]
        '''
        ent_hidden = x.clone()
        if mask_index is not None:
            ent_hidden[mask_index] = 0

        ent2sec_adj = ent2sec_mat
        ent2sec_sum = torch.sum(ent2sec_adj, 0).reshape(1, -1).repeat(ent2sec_adj.shape[0], 1)
        ent2sec_adj /= ent2sec_sum
        sector = torch.t(ent2sec_adj).mm(ent_hidden)
        
        sector = sector[sector.sum(1) != 0]
        ent2sec_adj = ent_hidden.mm(torch.t(sector))
        ent2sec_adj = self.softmax_along_0(ent2sec_adj)
        sector = torch.t(ent2sec_adj).mm(ent_hidden)

        sec2ent_adj_inv = ent_hidden.mm(torch.t(sector))
        sec2ent_adj_inv = self.softmax_along_1(sec2ent_adj_inv)

        ent_update = sec2ent_adj_inv.mm(sector)
        ent_update = self.fc_ent(ent_update)

        ent_skip_in = self.fc_skip_in(ent_update)
        ent_skip_out = self.fc_skip_out(ent_update)
        ent_skip_out = self.leaky_relu(ent_skip_out)

        if self.res_cof:
            ent_with_sec = x + self.res_cof_in * ent_skip_in
        else:
            ent_with_sec = x + ent_skip_in
        ent_to_gnn = ent_with_sec
        ent_to_gnn = self.fc_out_gnn(ent_to_gnn)
        ent_to_gnn = self.leaky_relu(ent_to_gnn)
        return (ent_skip_out, ent_to_gnn), self.res_cof_in
