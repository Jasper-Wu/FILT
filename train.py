from collections import defaultdict
import os
import time
import random
import argparse
import numpy as np 

from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

from TGEN_model import Induc

class Trainer(object):

    def __init__(self, args):

        super(Trainer, self).__init__()

        self.args = args
        self.exp_name, self.ts_name = self.experiment_name(args)
        self.logger = utils.setup_logger(self.exp_name, self.args.debug)
        self.logger.info(args)

        self.best_mrr = 0

        self.use_cuda = args.gpu >= 0 and torch.cuda.is_available()
        if self.use_cuda:
            torch.cuda.set_device(args.gpu)

        self.entity2id, self.relation2id, self.time2id, self.train_quads, self.valid_quads, self.test_quads, _ = utils.load_data_quadruples(f'./dataset/{args.data}')
        self.meta_train_task_entity_to_quads, self.meta_valid_task_entity_to_quads, self.meta_test_task_entity_to_quads = utils.load_processed_data_quadruples(f'./dataset/{args.data}/processed_data_{args.data_version}')

        self.all_quads = torch.LongTensor(np.concatenate((
            self.train_quads, self.valid_quads, self.test_quads
        )))

        self.meta_task_entity = np.concatenate((list(self.meta_valid_task_entity_to_quads.keys()),
                                            list(self.meta_test_task_entity_to_quads.keys())))
        self.meta_train_entity = np.array(list(self.meta_train_task_entity_to_quads.keys()))

        self.entities_list = np.delete(np.arange(len(self.entity2id)), self.meta_task_entity)

        self.load_pretrain_embedding()
        self.load_model()

        if self.use_cuda:
            self.model.cuda()
            self.all_quads = self.all_quads.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        if args.concept is not None:
            concept_mat = f'./dataset/{args.data}/auxiliary/ent2sec_matrix_v1.npy'
            self.ent2sec_matrix = torch.Tensor(np.load(concept_mat))
            print(f'loading matrix for concept from {concept_mat}')
            if self.use_cuda:
                self.ent2sec_matrix = self.ent2sec_matrix.cuda()

    def load_pretrain_embedding(self):

        self.embedding_size = int(self.args.pre_train_emb_size)

        if self.args.pre_train:

            pretrain_model_path = './pretrain/{}/{}'.format(self.args.data, self.args.pre_train)
            self.logger.info("load pre-train embedding from {}".format(pretrain_model_path))

            entity_file_name = os.path.join(pretrain_model_path, '{}_entity_{}.npy'.format(self.args.pre_train_model, self.embedding_size))
            relation_file_name = os.path.join(pretrain_model_path, '{}_relation_{}.npy'.format(self.args.pre_train_model, self.embedding_size))
            time_file_name = os.path.join(pretrain_model_path, '{}_time_{}.npy'.format(self.args.pre_train_model, self.embedding_size))

            self.pretrain_entity_embedding = torch.Tensor(np.load(entity_file_name))
            self.pretrain_relation_embedding = torch.Tensor(np.load(relation_file_name))
            self.logger.info("pretrain entity embedding shape: {}\npretrain relation embedding shape: {}".format(
                self.pretrain_entity_embedding.shape, self.pretrain_relation_embedding.shape))
            if os.path.exists(time_file_name):
                self.pretrain_time_embedding = torch.Tensor(np.load(time_file_name))
                self.logger.info("pretrain time embedding shape: {}".format(self.pretrain_time_embedding.shape))
            else:
                self.pretrain_time_embedding = None
            if self.pretrain_relation_embedding.shape[0] == 2 * len(self.relation2id) and not self.args.rev_rel_emb:
                self.logger.info(f"relation embedding: {self.pretrain_relation_embedding.shape[0]} -> {len(self.relation2id)}")
                self.pretrain_relation_embedding = (self.pretrain_relation_embedding[:len(self.relation2id)] + self.pretrain_relation_embedding[len(self.relation2id):]) / 2

        else:

            self.pretrain_entity_embedding = None
            self.pretrain_relation_embedding = None
            self.pretrain_time_embedding = None

    def load_model(self):

        if self.args.time_mode in ['tw']:
            self.model = Induc(self.embedding_size, self.embedding_size, self.embedding_size,
                                len(self.entity2id), len(self.relation2id), len(self.time2id),
                                args=self.args, entity_embedding=self.pretrain_entity_embedding,
                                relation_embedding=self.pretrain_relation_embedding,
                                time_embedding=self.pretrain_time_embedding, mode=self.args.time_mode)
        else:
            raise ValueError("Not supported time mode <{}>".format(self.args.time_mode))

        meta_task_entity = torch.LongTensor(self.meta_task_entity)
        meta_train_entity = torch.LongTensor(self.meta_train_entity)
        self.model.entity_embedding.weight.data[meta_task_entity] = torch.zeros(len(meta_task_entity), self.embedding_size)

    def train(self):

        start_epoch = 0

        self.logger.info("\nTraining...")
        cof_ll = []

        for epoch in trange(start_epoch, (self.args.n_epochs + 1), desc='Train Epochs', position=0):

            # Meta-Train
            self.model.train()
            
            train_task_pool = list(self.meta_train_task_entity_to_quads.keys())
            random.shuffle(train_task_pool)            

            total_loss = 0
            
            train_few = self.args.few

            unseen_entity_count = 0
            temp_loss = 0
            cof_list = []
            
            if self.args.concept:
                concept_ent_emb, res_cof_in = self.model.concept_model(self.model.entity_embedding.weight, self.ent2sec_matrix)
                res_cof_in = res_cof_in.item() if isinstance(res_cof_in, nn.parameter.Parameter) else res_cof_in
            else:
                concept_ent_emb = None

            for unseen_entity in train_task_pool[:self.args.num_train_entity]:

                quads = self.meta_train_task_entity_to_quads[unseen_entity]
                random.shuffle(quads)
                quads = np.array(quads)
                quads = quads[quads[:, 0] != quads[:, 2]]

                heads, relations, tails, ts = quads.transpose()

                train_quads = quads[:train_few]
                test_quads = quads[train_few:]

                if (len(quads)) - train_few < 1:
                    continue

                entities_list = self.entities_list
                false_candidates = np.setdiff1d(entities_list, np.concatenate([heads, tails]))
                false_entities = np.random.choice(false_candidates, size=(len(quads) - train_few) * self.args.negative_sample)

                pos_samples = test_quads
                query_t = test_quads[:, 3]
                query_r = test_quads[:, 1]
                neg_samples = np.tile(pos_samples, (self.args.negative_sample, 1))
                neg_samples[neg_samples[:, 0] == unseen_entity, 2] = false_entities[neg_samples[:, 0] == unseen_entity]
                neg_samples[neg_samples[:, 2] == unseen_entity, 0] = false_entities[neg_samples[:, 2] == unseen_entity]
                samples = np.concatenate((pos_samples, neg_samples))
                samples = torch.LongTensor(samples)

                if self.use_cuda:
                    samples = samples.cuda()

                # Train & loss
                if self.args.time_mode == 'tw':
                    unseen_entity_embedding, res_cof_out = self.model.forward_tw(unseen_entity, train_quads, query_t, self.use_cuda, query_r, cover_ent_emb=concept_ent_emb)
                    loss = self.model.score_loss_td_batch(unseen_entity, unseen_entity_embedding, samples, self.use_cuda)
                    res_cof_out = res_cof_out.item() if isinstance(res_cof_out, nn.parameter.Parameter) else res_cof_out
                    cof_list.append(res_cof_out)

                total_loss += loss.item()
                temp_loss += loss
                unseen_entity_count += 1

            temp_loss /= unseen_entity_count
            temp_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

            cof_ll.append(cof_list)

            avg_loss = total_loss / self.args.num_train_entity

            # Meta-Valid
            if epoch % self.args.evaluate_every == 0:

                self.logger.info("Epochs-{}, Loss-{}".format(epoch, avg_loss))
                if self.args.res_cof:
                    self.logger.info("epoch {}: residual coefficient {:.4g}, {:.4g}".format(epoch, res_cof_in, np.mean([np.mean(x) for x in cof_ll[-self.args.evaluate_every:]])))

                with torch.no_grad():

                    results = self.eval(eval_type='valid')

                    mrr = results['total_mrr']
                    self.logger.info("Total MRR (filtered): {:.6f}".format(results['total_mrr']))
                    self.logger.info("Total Hits (filtered) @ {}: {:.6f}".format(1, results['total_hits@1']))
                    self.logger.info("Total Hits (filtered) @ {}: {:.6f}".format(3, results['total_hits@3']))
                    self.logger.info("Total Hits (filtered) @ {}: {:.6f}".format(10, results['total_hits@10']))

                if mrr > self.best_mrr:
                    self.best_mrr = mrr
                    if not self.args.debug:
                        torch.save({'model_state_dict': self.model.state_dict(),
                                    'optimizer_state_dict': self.optimizer.state_dict(),
                                    'epoch': epoch,
                                    'params': self.args},
                                './checkpoints/{}/best_mrr_model.pth'.format(self.ts_name))

        self.logger.info("\nTesting...")

        if not self.args.debug:
            checkpoint = torch.load('./checkpoints/{}/best_mrr_model.pth'.format(self.ts_name))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info("Using best epoch: {}, {}".format(checkpoint['epoch'], self.exp_name))

        # Meta-Test
        with torch.no_grad():
            results = self.eval(eval_type='test')

        self.logger.info("Total MRR (filtered): {:.6f}".format(results['total_mrr']))
        self.logger.info("Total Hits (filtered) @ {}: {:.6f}".format(1, results['total_hits@1']))
        self.logger.info("Total Hits (filtered) @ {}: {:.6f}".format(3, results['total_hits@3']))
        self.logger.info("Total Hits (filtered) @ {}: {:.6f}".format(10, results['total_hits@10']))

    def eval(self, eval_type='test'):

        self.model.eval()

        if eval_type == 'valid':
            test_task_dict = self.meta_valid_task_entity_to_quads
            test_task_pool = list(self.meta_valid_task_entity_to_quads.keys())
        elif eval_type == 'test':
            test_task_dict = self.meta_test_task_entity_to_quads
            test_task_pool = list(self.meta_test_task_entity_to_quads.keys())
        else:
            raise ValueError("Eval Type <{}> is Wrong".format(eval_type))

        total_ranks = []
        total_subject_ranks = []
        total_object_ranks = []

        if self.args.concept:
            concept_ent_emb, res_cof_in = self.model.concept_model(self.model.entity_embedding.weight, self.ent2sec_matrix)
        else:
            concept_ent_emb = None

        for unseen_entity in test_task_pool:

            quads = test_task_dict[unseen_entity]
            quads = np.array(quads)
            heads, relations, tails, ts = quads.transpose()

            train_quads = quads[:self.args.few]
            test_quads = quads[self.args.few:]
            query_t = test_quads[:, 3]
            query_r = test_quads[:, 1]

            if (len(quads)) - self.args.few < 1:
                continue

            test_quads = torch.LongTensor(test_quads)
            if self.use_cuda:
                test_quads = test_quads.cuda()

            if self.args.time_mode == 'tw':
                unseen_entity_embedding, _ = self.model.forward_tw(unseen_entity, train_quads, query_t, self.use_cuda, query_r, cover_ent_emb=concept_ent_emb)
                ranks, ranks_s, ranks_o = utils.calc_induc_mrr_batch_head(unseen_entity, unseen_entity_embedding,
                                                    self.model.entity_embedding.weight, self.model.relation_embedding, test_quads,
                                                    self.all_quads, self.use_cuda, score_function=self.args.score_function, inv_rel=self.args.rev_rel_emb)
            if len(ranks_s) != 0:
                total_subject_ranks.append(ranks_s)

            if len(ranks_o) != 0:
                total_object_ranks.append(ranks_o)

            total_ranks.append(ranks)

        results = {}
                        
        # Subject
        total_subject_ranks = torch.cat(total_subject_ranks)
        total_subject_ranks += 1

        results['subject_ranks'] = total_subject_ranks
        results['subject_mrr'] = torch.mean(1.0 / total_subject_ranks.float()).item()

        for hit in [1, 3, 10]:
            avg_count = torch.mean((total_subject_ranks <= hit).float())
            results['subject_hits@{}'.format(hit)] = avg_count.item()

        # Object
        total_object_ranks = torch.cat(total_object_ranks)
        total_object_ranks += 1

        results['object_ranks'] = total_object_ranks
        results['object_mrr'] = torch.mean(1.0 / total_object_ranks.float()).item()

        for hit in [1, 3, 10]:
            avg_count = torch.mean((total_object_ranks <= hit).float())
            results['object_hits@{}'.format(hit)] = avg_count.item()

        # Total
        total_ranks = torch.cat(total_ranks)
        total_ranks += 1

        results['total_ranks'] = total_ranks
        results['total_mrr'] = torch.mean(1.0 / total_ranks.float()).item()

        for hit in [1, 3, 10]:
            avg_count = torch.mean((total_ranks <= hit).float())
            results['total_hits@{}'.format(hit)] = avg_count.item()

        return results

    def experiment_name(self, args):

        ts = time.strftime('%Y%b%d-%H%M%S', time.gmtime())

        exp_name = f'{ts}_{args.data}_{args.data_version}_{args.time_mode}_{args.few}-shot'

        if not args.debug:
            if not(os.path.isdir('./checkpoints/{}'.format(ts))):
                os.makedirs(os.path.join('./checkpoints/{}'.format(ts)))

            print("Make directory {} in a checkpoints folder".format(ts))

        return exp_name, ts

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='FILT')
    
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action='store_true')

    parser.add_argument("--data", type=str, default="ICEWS14")
    parser.add_argument("--data-version", type=str, default='v1')
    parser.add_argument("--negative-sample", type=int, default=32)

    parser.add_argument("--few", type=int, default=3)
    parser.add_argument("--n-epochs", type=int, default=15000)
    parser.add_argument("--evaluate-every", type=int, default=100)

    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", default=0.0, type=float)

    parser.add_argument("--grad-norm", type=float, default=1.0)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument("--pre-train", type=str, default=None)  # folder suffix, e.g. v0
    parser.add_argument("--fine-tune", action='store_true')

    parser.add_argument("--pre-train-model", type=str, default='ComplEx')
    parser.add_argument("--pre-train-emb-size", type=str, default='100')
    parser.add_argument("--num-train-entity", type=int, default=100)
    parser.add_argument("--score-function", type=str, default='ComplEx')

    parser.add_argument("--time-mode", type=str)    # 'tw'
    parser.add_argument("--res-cof", type=float, default=0.0)
    parser.add_argument("--rev-rel-emb", action="store_true")
    parser.add_argument("--concept", action='store_true')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    trainer = Trainer(args)
    trainer.train()
    # print(args)

# python train.py --data ICEWS14 --data-version v0 --pre-train v0 --time-mode tw --fine-tune --rev-rel-emb --gpu 0 --few 3 --concept --res-cof 0.1 --debug