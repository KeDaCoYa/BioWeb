# -*- encoding: utf-8 -*-
"""
@File    :   train.py   
@Contact :   1329818994@qq.com
@License :   (C)Copyright 2022-2099
@Author  :   Kedaxia
@Version :   0.0.1
@Create time :   2022/4/6 14:18   
@Description :   None 

"""
import logging
import pickle

from src.norm.utils.dataset_utils import DictionaryDataset

logger = logging.getLogger("main.norm.train_utils")
def load_new_dictionary(biosyn,dictionary_path):
    """
    这是根据已有的字典文件来生成所需的向量形式
    """
    logger.info("开始对字典:{}进行初次初始化".format(dictionary_path))
    dictionary = DictionaryDataset(dictionary_path=dictionary_path).data
    dictionary_names = dictionary[:, 0]
    dict_sparse_embeds = biosyn.get_sparse_representation(mentions=dictionary_names, verbose=True)
    dict_dense_embeds = biosyn.get_dense_representation(mentions=dictionary_names, verbose=True)

    return dictionary, dict_sparse_embeds, dict_dense_embeds

def load_cache_dictionary(cahe_path):
    """
    加载已经经过
    """
    with open(cahe_path, 'rb') as fin:
        cached_dictionary = pickle.load(fin)
    logger.info("加载已有的cache dictioanry {}".format(cahe_path))

    dictionary, dict_sparse_embeds, dict_dense_embeds = (
        cached_dictionary['dictionary'],
        cached_dictionary['dict_sparse_embeds'],
        cached_dictionary['dict_dense_embeds'],
    )
    return dictionary, dict_sparse_embeds, dict_dense_embeds
