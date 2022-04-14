# -*- encoding: utf-8 -*-
"""
@File    :   norm_predicate.py   
@Contact :   1329818994@qq.com
@License :   (C)Copyright 2022-2099
@Author  :   Kedaxia
@Version :   0.0.1
@Create time :   2022/4/6 8:45   
@Description :   None 

"""
import os
import logging
import pickle

import torch
from ipdb import set_trace

from src.norm.config import MyBertConfig
from src.norm.models.biosyn import BioSyn
from src.norm.utils.dataset_utils import DictionaryDataset

logger = logging.getLogger("main.norm.predicate")
def get_norm_bert_config():
    """
    这里手动初始化在predicate时候需要的各种参数
    """
    config = MyBertConfig(
        bert_dir="../../embedding/SapBERT-from-PubMedBERT-fulltext",
        bert_name='biobert',
        use_gpu=True if torch.cuda.is_available() else False,
        max_len=25,
        batch_size=32,
        seed=1234,
        gpu_id=0,
        logfile_name='log',
        verbose=True,


    )
    return config

def init_dictionary(config):
    """
    将多个字典加载出来(除了gene)
    """
    pass
def init_norm_model(config,ckpt_path):
    # load biosyn model
    device = torch.device('cuda') if config.use_gpu and torch.cuda.is_available() else torch.device('cpu')

    biosyn = BioSyn(config, device)
    # 加载已经训练的模型,dense_encoder,sparse_encoder

    biosyn.load_model(model_name_or_path=ckpt_path)


    return biosyn, device