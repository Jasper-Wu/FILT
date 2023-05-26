# FILT
code for AKBC2022 paper *Few-Shot Inductive Learning on Temporal Knowledge Graphs using Concept-Aware Information*

## Pretrain Embedding

- We provide the ComplEx pretrain embedding for ICEWS14-OOG, ICEWS0515-OOG, ICEWS18-OOG in https://drive.google.com/file/d/1aUemsoBCF7FOFgM_V52-beFzwWU6ztBo/view?usp=sharing

## Installation
Create a conda environment

```
conda create -n filt python=3.7
conda activate filt
```

Configure FILT requirements

Install PyTorch
```
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```
Install PyTorch Geometric
```
conda install pyg -c pyg
```

## Datasets Generation

The preprocessed ICEWS14 dataset is already in the repository, including the raw dataset and the dataset for meta-learning
- raw dataset: already mapped to integers
- meta-learning dataset: stored in `./dataset/ICEWS14/processed_data/`
- the script for generating the meta-learning dataset is './data_preprocess.py'

In addition, you have to generate the entity-to-sector matrix with the following command
```
python generate_ent2sec_mat.py
```

## Training Model

Training  FILT on ICEWS14:

```
python train.py --data ICEWS14 --time-mode tw --fine-tune --rev-rel-emb --few 3 --concept --res-cof 0.1 --gpu 0
```
