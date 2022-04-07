# -*- encoding: utf-8 -*-
"""
@File    :   model_utils.py   
@Contact :   1329818994@qq.com
@License :   (C)Copyright 2022-2099
@Author  :   Kedaxia
@Version :   0.0.1
@Create time :   2022/4/4 21:18   
@Description :   None 

"""
import logging

import torch
from ipdb import set_trace

logger = logging.getLogger('ner.model_utils')

def load_model(model, ckpt_path=None):
    """
    加载模型 & 放置到 GPU 中（单卡 / 多卡）
    load_type:表示加载模型的类别，one2one,one2many,many2one,many2many
    """

    if ckpt_path is not None:

        checkpoint = torch.load(ckpt_path)
        try:
            model.load_state_dict(checkpoint)
        except:
            logger.info('这是多GPU训练得到的模型，重新修改')
            from collections import OrderedDict
            new_checkpoint = OrderedDict()
            for key,value in checkpoint.items():

                new_key = ".".join(key.split('.')[1:])
                new_checkpoint[new_key] = value
            try:
                model.load_state_dict(new_checkpoint)
            except:

                embedding_shape = checkpoint['bert_model.embeddings.word_embeddings.weight'].shape
                model.bert_model.resize_token_embeddings(embedding_shape[0])
                model.load_state_dict(checkpoint)
                logger.warning("发生异常，重新调整word embedding size加载成功")

    return model



def span_predicate(start_logits,end_logits,raw_text_li,span_id2label):
    '''
    一般只有bert采用这个span，使用bilstm，这个span没啥意思
    :param config:
    :param best_model_ckpt_path:
    :param kwargs:
    :return:
    '''
    batch_size = len(raw_text_li)


    entities = []
    for id,length in enumerate(range(batch_size)):
        # 这里是一个集合，共由四部分组成(entity,entity_type,start_idx,end_idx)
        tmp_start_logits = start_logits[id]
        tmp_end_logits = end_logits[id]


        for i,s_type in enumerate(tmp_start_logits):
            if s_type == 0: # 忽略Other 的标签
                continue
            for j,e_type in enumerate(tmp_end_logits[i:]):
                if s_type == e_type:

                    entities.append({
                    'entity_type': span_id2label[s_type],
                    'start_idx': str(i),
                    'end_idx': str(i+j),
                    'entity_name': " ".join(raw_text_li[id][i:i+j+1])
                    })

                    break

    return entities

