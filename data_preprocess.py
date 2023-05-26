from utils import *
import os
import pickle
import random
import numpy as np
from collections import Counter

dataset_path = './dataset/ICEWS14'
lower_bound_to_select_task_entity = 10
upper_bound_to_select_task_entity = 25
meta_train_ratio = 0.8
meta_valid_ratio = 0.1
meta_test_ratio = 0.1
meta_task_ratio = 0.5

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

ent2id, rel2id, time2id, train_quads, valid_quads, test_quads, all_quads = load_data_quadruples(dataset_path)
np.random.shuffle(all_quads)

head, relation, tail, timestamp = all_quads.transpose()

entity = np.concatenate([head, tail])
entity_counter = Counter(entity)
entity_counter_values_counter = Counter(entity_counter.values())
low_frequency_entity_list = []

for entity_id in entity_counter:
    entity_frequency = entity_counter[entity_id]
    if (entity_frequency >= lower_bound_to_select_task_entity) and (entity_frequency <= upper_bound_to_select_task_entity):
        low_frequency_entity_list.append(entity_id)

n_low_freq_entity = len(low_frequency_entity_list)
n_sampled_entity = int(n_low_freq_entity * meta_task_ratio)  # 0.5
meta_train_number = int(meta_train_ratio * n_sampled_entity)  # 0.8
meta_valid_number = int(meta_valid_ratio * n_sampled_entity)  # 0.1
meta_test_number = int(meta_test_ratio * n_sampled_entity)
print("num_low_frequency_entity: {}".format(n_low_freq_entity))
print("sampled_num_low_frequency_entity: {}".format(n_sampled_entity))
print(meta_train_number, meta_valid_number, meta_test_number)

sampled_low_frequency_entity_list = np.random.choice(low_frequency_entity_list, n_sampled_entity, replace=False)

filtered_index = []
for index, quadruple in enumerate(all_quads):
    if quadruple[0] in sampled_low_frequency_entity_list:
        continue
    elif quadruple[2] in sampled_low_frequency_entity_list:
        continue
    filtered_index.append(index)  # index for quadruples that don't have unseen_entity

filtered_index = np.array(filtered_index)
filtered_quadruples = all_quads[filtered_index]

meta_train_entity_list = sampled_low_frequency_entity_list[:meta_train_number]
meta_valid_entity_list = sampled_low_frequency_entity_list[meta_train_number:meta_train_number + meta_valid_number]
meta_test_entity_list = sampled_low_frequency_entity_list[meta_train_number + meta_valid_number:]

def process_meta_set(index, quadruple, self_list, other_list,
                     idx_list, cross_idx_list, self_dict, self_cross_dict):
    '''
    params:
        self_list: meta_train_entity_list
        other_list: (meta_valid_entity_list, meta_test_entity_list)
        self_dict: meta_train_task_entity_to_quadruples
        self_cross_dict: meta_train_task_entity_to_unseen_quadruples
        idx_list: meta_train_task_index
        cross_idx_list: meta_train_unseen_task_index
    global:
        meta_train_task_index
        meta_train_unseen_task_index
    '''
    if (quadruple[0] in self_list) or (quadruple[2] in self_list):
        if (quadruple[0] not in other_list[0]) and (quadruple[2] not in other_list[0]) \
                and (quadruple[0] not in other_list[1]) and (quadruple[2] not in other_list[1]):

            idx_list.append(index)

            # quadruples between two meta_train entity
            if (quadruple[0] in self_list) and (quadruple[2] in self_list):
                cross_idx_list.append(index)
                i = random.randrange(0, 4, 2)
                # randomly add to one of the unseen entity in the dictionary
                if quadruple[i] in self_cross_dict:
                    self_cross_dict[quadruple[i]].append(quadruple)
                else:
                    self_cross_dict[quadruple[i]] = [quadruple]
            # quadruples between meta_train entity to in-graph entity
            elif (quadruple[0] in self_list):
                i = 0
            elif (quadruple[2] in self_list):
                i = 2

            if quadruple[i] in self_dict:
                self_dict[quadruple[i]].append(quadruple)
            else:
                self_dict[quadruple[i]] = [quadruple]

meta_train_task_index = []  # index for quadruples: meta_train entity to in-graph & other meta_train
meta_valid_task_index = []
meta_test_task_index = []

meta_train_task_entity_to_quadruples = {}
meta_valid_task_entity_to_quadruples = {}
meta_test_task_entity_to_quadruples = {}

meta_train_unseen_task_index = []  # quadruples between meta_train entity and other meta_train entity
meta_valid_unseen_task_index = []
meta_test_unseen_task_index = []

meta_train_task_entity_to_unseen_quadruples = {}
meta_valid_task_entity_to_unseen_quadruples = {}
meta_test_task_entity_to_unseen_quadruples = {}

for index, quadruple in enumerate(all_quads):
    # Meta-Train Quadruples
    # quadruple include meta_train_entity
    process_meta_set(index, quadruple, meta_train_entity_list, (meta_valid_entity_list, meta_test_entity_list),
                     meta_train_task_index, meta_train_unseen_task_index,
                     meta_train_task_entity_to_quadruples, meta_train_task_entity_to_unseen_quadruples)

    # Meta-Valid quadruple
    process_meta_set(index, quadruple, meta_valid_entity_list, (meta_train_entity_list, meta_test_entity_list),
                     meta_valid_task_index, meta_valid_unseen_task_index,
                     meta_valid_task_entity_to_quadruples, meta_valid_task_entity_to_unseen_quadruples)

    process_meta_set(index, quadruple, meta_test_entity_list, (meta_train_entity_list, meta_valid_entity_list),
                     meta_test_task_index, meta_test_unseen_task_index,
                     meta_test_task_entity_to_quadruples, meta_test_task_entity_to_unseen_quadruples)

meta_train_task_quadruples = all_quads[meta_train_task_index]
meta_valid_task_quadruples = all_quads[meta_valid_task_index]
meta_test_task_quadruples = all_quads[meta_test_task_index]

meta_train_unseen_task_quadruples = all_quads[meta_train_unseen_task_index]
meta_valid_unseen_task_quadruples = all_quads[meta_valid_unseen_task_index]
meta_test_unseen_task_quadruples = all_quads[meta_test_unseen_task_index]

print('num_meta_train_task_quadruples: {}'.format(len(meta_train_task_quadruples)))
print('num_meta_valid_task_quadruples: {}'.format(len(meta_valid_task_quadruples)))
print('num_meta_test_task_quadruples: {}'.format(len(meta_test_task_quadruples)))

print('num_meta_train_entity: {}'.format(len(meta_train_task_entity_to_quadruples.keys())))
print('num_meta_valid_entity: {}'.format(len(meta_valid_task_entity_to_quadruples.keys())))
print('num_meta_test_entity: {}'.format(len(meta_test_task_entity_to_quadruples.keys())))

print('num_meta_train_unseen_task_quadruples: {}'.format(len(meta_train_unseen_task_quadruples)))
print('num_meta_valid_unseen_task_quadruples: {}'.format(len(meta_valid_unseen_task_quadruples)))
print('num_meta_test_unseen_task_quadruples: {}'.format(len(meta_test_unseen_task_quadruples)))

print('num_meta_train_unseen_entity_pair: {}'.format(len(meta_train_task_entity_to_unseen_quadruples.keys())))
print('num_meta_valid_unseen_entity_pair: {}'.format(len(meta_valid_task_entity_to_unseen_quadruples.keys())))
print('num_meta_test_unseen_entity_pair: {}'.format(len(meta_test_task_entity_to_unseen_quadruples.keys())))


count_lowfreq = 0
count_len = 0
for task_entity, quadruples in meta_valid_task_entity_to_quadruples.items():
    count_len += len(quadruples)
    if len(quadruples) < 2:
        count_lowfreq += 1
print("number of quadruples in entity_to_quadruples dictionary: {}".format(count_len))
print("number of entity with less than 2 quadruples in meta_valid: {}".format(count_lowfreq))

count_lowfreq = 0
count_len = 0
for task_entity, quadruples in meta_test_task_entity_to_quadruples.items():
    count_len += len(quadruples)
    if len(quadruples) < 2:
        count_lowfreq += 1
print("number of quadruples in entity_to_quadruples dictionary: {}".format(count_len))
print("number of entity with less than 2 quadruples in meta_test: {}".format(count_lowfreq))


all_quadruples_to_use = np.concatenate(
    (filtered_quadruples, meta_train_task_quadruples, meta_valid_task_quadruples, meta_test_task_quadruples))
# print('num_all_quadruples: {}'.format(len(all_quadruples_to_use)))

save_folder = './dataset/ICEWS14/processed_data/'
os.makedirs(save_folder, exist_ok=True)

with open(save_folder + 'filtered_quadruples.pickle', 'wb') as f:
    pickle.dump(filtered_quadruples, f)
with open(save_folder + 'meta_train_task_entity_to_quadruples.pickle', 'wb') as f:
    pickle.dump(meta_train_task_entity_to_quadruples, f)
with open(save_folder + 'meta_valid_task_entity_to_quadruples.pickle', 'wb') as f:
    pickle.dump(meta_valid_task_entity_to_quadruples, f)
with open(save_folder + 'meta_test_task_entity_to_quadruples.pickle', 'wb') as f:
    pickle.dump(meta_test_task_entity_to_quadruples, f)
print(f"save processed data in folder {save_folder}")
