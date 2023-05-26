import os
import pickle
import logging
import numpy as np

import torch

################################################
#   Data Load and Pre-process for quadruples   #
################################################

def load_data_quadruples(file_path):
    print("load data from {}".format(file_path))

    with open(os.path.join(file_path, 'entity2id.txt'), encoding='utf-8') as f:
        entity2id = dict()
        for line in f:
            entity, eid = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(file_path, 'relation2id.txt'), encoding='utf-8') as f:
        relation2id = dict()
        for line in f:
            relation, rid = line.strip().split('\t')
            relation2id[relation] = int(rid)

    time_file = os.path.join(file_path, 'time2id.txt')
    time2id = None
    if os.path.exists(time_file):
        with open(time_file, 'r', encoding='utf-8') as f:
            time2id = dict()
            for line in f:
                time_str, tid = line.strip().split('\t')
                time2id[time_str] = int(tid)

    train_quadruples = read_quadruples(os.path.join(file_path, 'train.txt'), entity2id, relation2id)
    valid_quadruples = read_quadruples(os.path.join(file_path, 'valid.txt'), entity2id, relation2id)
    test_quadruples = read_quadruples(os.path.join(file_path, 'test.txt'), entity2id, relation2id)
    all_quadruples = np.concatenate([train_quadruples, valid_quadruples, test_quadruples], axis=0)

    print('num_entity: {}'.format(len(entity2id)))
    print('num_relation: {}'.format(len(relation2id)))
    print('num_train_quads: {}'.format(len(train_quadruples)))
    print('num_valid_quads: {}'.format(len(valid_quadruples)))
    print('num_test_quads: {}'.format(len(test_quadruples)))
    print('num_all_quads: {}'.format(len(all_quadruples)))
    print("finish loading raw data!\n")

    if time2id is None:
        return entity2id, relation2id, None, train_quadruples, valid_quadruples, test_quadruples, all_quadruples
    else:
        return entity2id, relation2id, time2id, train_quadruples, valid_quadruples, test_quadruples, all_quadruples

def read_quadruples(file_path, entity2id=None, relation2id=None, time2id=None):
    quadruples = []
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            l = line.strip().split('\t')
            if len(l) == 4:
                head, rel, tail, t = l
            elif len(l) == 5:
                head, rel, tail, t, _ = l
            if entity2id and relation2id and time2id:
                quadruples.append((entity2id[head], relation2id[rel], entity2id[tail], time2id[t]))
            else:
                quadruples.append((int(head), int(rel), int(tail), int(t)))
    return np.array(quadruples)

def load_processed_data_quadruples(file_path):

    print(f"load processed data from {file_path}")
    with open(os.path.join(file_path, 'meta_train_task_entity_to_quadruples.pickle'), 'rb') as f:
        meta_train_task_entity_to_quadruples = pickle.load(f)

    with open(os.path.join(file_path, 'meta_valid_task_entity_to_quadruples.pickle'), 'rb') as f:
        meta_valid_task_entity_to_quadruples = pickle.load(f)

    with open(os.path.join(file_path, 'meta_test_task_entity_to_quadruples.pickle'), 'rb') as f:
        meta_test_task_entity_to_quadruples = pickle.load(f)

    print(f"num of train entity: {len(meta_train_task_entity_to_quadruples)}\n"
          f"num of total train quadruples: {np.sum([len(v) for v in meta_train_task_entity_to_quadruples.values()])}\n"
          f"num of valid entity: {len(meta_valid_task_entity_to_quadruples)}\n"
          f"num of total valid quadruples: {np.sum([len(v) for v in meta_valid_task_entity_to_quadruples.values()])}\n"
          f"num of test entity: {len(meta_test_task_entity_to_quadruples)}\n"
          f"num of total test quadruples: {np.sum([len(v) for v in meta_test_task_entity_to_quadruples.values()])}")

    return meta_train_task_entity_to_quadruples, meta_valid_task_entity_to_quadruples, meta_test_task_entity_to_quadruples

#########################
#   Cal Score (Ranks)   #
#########################

def sort_and_rank(score, target):
    '''
    input:
        score: [1, -1], [1, N]
    return:
        order of target in predicted score
    question:

    '''
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1), as_tuple=False)
    indices = indices[:, 1].view(-1)
    return indices

def complex_score(head_embeddings, relation_embeddings, tail_embeddings):
    dim = head_embeddings.shape[1]
    h_emb = head_embeddings[:, :dim // 2], head_embeddings[:, dim // 2:]
    rel_emb = relation_embeddings[:, :dim // 2], relation_embeddings[:, dim // 2:]
    t_emb = tail_embeddings[:, :dim // 2], tail_embeddings[:, dim // 2:]
    score = torch.sum((h_emb[0] * rel_emb[0] - h_emb[1] * rel_emb[1]) * t_emb[0] +
                      (h_emb[0] * rel_emb[1] + h_emb[1] * rel_emb[0]) * t_emb[1], dim=1)
    return score

def get_delete_index(fact, kg, temporal=True):
    '''
    Args:
        fact: [sub, rel, ts] or [obj, rel, ts]
    '''
    if temporal:
        delete_count = torch.sum(fact == kg, dim=1)
        delete_index = torch.nonzero(delete_count==3, as_tuple=False).squeeze()
    else:
        delete_count = torch.sum(fact[:2] == kg[:, :2], dim=1)
        delete_index = torch.nonzero(delete_count==2, as_tuple=False).squeeze()
    return delete_index

def get_candidate_index(triple, all_quads, num_entity, mode, filtered, use_cuda):
    '''
    mode: 'rhs' for unseen as subject or 'lhs' for unseen as object
    '''
    device = torch.device('cuda') if use_cuda else None
    if mode == 'sub':
        head_rel_ts = all_quads[:, [0, 1, 3]]
        srt = triple[[0, 1, 3]]
        cand_pos = 2
    elif mode == 'obj':
        head_rel_ts = all_quads[:, [2, 1, 3]]
        srt = triple[[2, 1, 3]]
        cand_pos = 0
    delete_index = get_delete_index(srt, head_rel_ts, temporal=False)
    delete_entity_index = all_quads[delete_index, cand_pos].view(-1)
    delete_entity_index = delete_entity_index.cpu().numpy() if use_cuda else delete_entity_index.numpy()
    perturb_entity_index = np.setdiff1d(np.arange(num_entity), delete_entity_index)
    perturb_entity_index = torch.from_numpy(perturb_entity_index).to(device) if use_cuda else torch.from_numpy(
        perturb_entity_index)
    perturb_entity_index = torch.cat([triple[cand_pos].view(-1), perturb_entity_index])
    return perturb_entity_index

def calc_induc_mrr_batch_head(unseen_entity, unseen_entity_embedding_batch, all_entity_embeddings, all_relation_embeddings,
                            test_quads, all_quads, use_cuda, score_function="DistMult", inv_rel=False, filtered=None,
                            time_embeddings=None, time_gran=24):
    assert len(unseen_entity_embedding_batch) == len(test_quads)
    num_entity = len(all_entity_embeddings)
    subject_count = 0
    object_count = 0

    ranks = []
    ranks_s = []
    ranks_o = []

    head_relation_ts = all_quads[:, [0, 1, 3]]
    tail_relation_ts = all_quads[:, [2, 1, 3]]

    device = torch.device('cuda') if use_cuda else None

    for i, test_triplet in enumerate(test_quads):
        unseen_entity_embedding = unseen_entity_embedding_batch[i]
        ts = torch.div(test_triplet[3], time_gran, rounding_mode='floor')

        if test_triplet[0] == unseen_entity:
            mode = 'sub'
            subject_count += 1
            relation = test_triplet[1]

        elif test_triplet[2] == unseen_entity:
            mode = 'obj'
            object_count += 1
            if not inv_rel:
                relation = test_triplet[1]
            else:
                relation = test_triplet[1] + len(all_relation_embeddings) // 2

        perturb_entity_index = get_candidate_index(test_triplet, all_quads, num_entity,
                                    mode=mode, filtered=filtered, use_cuda=use_cuda)

        # Score
        if score_function == 'ComplEx':
            score = complex_score(unseen_entity_embedding.view(1, -1), all_relation_embeddings[[relation], :],
                                  all_entity_embeddings[perturb_entity_index, :]).view(1, -1)
        else:
            raise TypeError

        # Cal Rank
        target = torch.tensor(0).to(device) if use_cuda else torch.tensor(0)
        if mode == 'sub':
            ranks_s.append(sort_and_rank(score, target))
        elif mode == 'obj':
            ranks_o.append(sort_and_rank(score, target))

    if subject_count == 0:
        ranks_o = torch.cat(ranks_o)
        ranks = ranks_o

    elif object_count == 0:
        ranks_s = torch.cat(ranks_s)
        ranks = ranks_s

    else:
        ranks_s = torch.cat(ranks_s)
        ranks_o = torch.cat(ranks_o)
        ranks = torch.cat((ranks_s, ranks_o))

    return ranks, ranks_s, ranks_o


def setup_logger(name, debug=False):
    cur_dir = os.getcwd()
    if not os.path.exists(cur_dir + '/log/'):
        os.mkdir(cur_dir + '/log/')
    
    logger_name = 'log'
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    fmt = "%(asctime)s %(name)s %(levelname)-8s %(message)s"
    datefmt = "%m-%d %H:%M"
    formatter = logging.Formatter(fmt, datefmt)
    if not debug:
        filename = cur_dir + '/log/' + name + '.log'
        fh = logging.FileHandler(filename, mode='a')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.info("log file: {}.log".format(name))

    sh = logging.StreamHandler()
    logger.addHandler(sh)
    return logger

def read_dict(file_path):
    k2v = {}
    v2k = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for l in f:
            l = l.strip().split('\t')
            name, idx = l[0], int(l[1])
            k2v[name] = idx
            v2k[idx] = name
    return k2v, v2k
