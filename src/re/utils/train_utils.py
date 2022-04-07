# -*- encoding: utf-8 -*-
"""
@File    :   train_utils.py   
@Contact :   1329818994@qq.com
@License :   (C)Copyright 2022-2099
@Author  :   Kedaxia
@Version :   0.0.1
@Create time :   2022/4/5 15:02   
@Description :   None 

"""
import logging

import numpy as np
import torch

from src.re.model.mtb_bert import MTBRelationClassification
from src.re.model.mul_bert import MulBERT
from src.re.model.rbert import RBERT

logger = logging.getLogger('main.re.train_utils')

def relation_classification_decode(logits):
    '''
    这里是解码，将关系分类预测的结果进行解码
    :param logits: shape=(batch_size,num_labels )
    :return:
    '''
    output = np.argmax(logits.detach().cpu().numpy(), axis=1)
    return output
def set_tokenize_special_tag(config,tokenizer):

    tokenizer.add_tokens(config.ent1_start_tag)
    tokenizer.add_tokens(config.ent1_end_tag)
    tokenizer.add_tokens(config.ent2_start_tag)
    tokenizer.add_tokens(config.ent2_end_tag)

    config.ent1_start_tag_id = tokenizer.convert_tokens_to_ids(config.ent1_start_tag)
    config.ent1_end_tag_id = tokenizer.convert_tokens_to_ids(config.ent1_end_tag)
    config.ent2_start_tag_id = tokenizer.convert_tokens_to_ids(config.ent2_start_tag)
    config.ent2_end_tag_id = tokenizer.convert_tokens_to_ids(config.ent2_end_tag)

def choose_re_model(config):
    if config.model_name == 'r_bert':
        model = RBERT(config)
    elif config.model_name == 'mul_bert':
        model = MulBERT(config)
    elif config.model_name == 'mtb_bert':
        model = MTBRelationClassification(config)
    elif config.model_name == 'kebiolm':
        # todo : 补充kebiolm的模型方法
        pass
    else:

        raise ValueError
    return model


def load_model_and_parallel(model, gpu_ids, ckpt_path=None, strict=True, load_type='one2one'):
    """
    加载模型 & 放置到 GPU 中（单卡 / 多卡）
    主要是多卡模型训练和保存的问题...
    load_type:表示加载模型的类别，one2one,one2many,many2one,many2many
    """
    gpu_ids = gpu_ids.split(',')
    device = torch.device("cpu" if gpu_ids[0] == '-1' else "cuda:" + gpu_ids[0])

    if load_type == 'one2one':
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            try:

                model.load_state_dict(checkpoint)
            except:
                embedding_shape = checkpoint['bert_model.embeddings.word_embeddings.weight'].shape
                model.bert_model.resize_token_embeddings(embedding_shape[0])
                model.load_state_dict(checkpoint)
                logger.warning("发生异常，重新调整word embedding size加载成功")



    elif load_type == 'one2many':
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            model.load_state_dict(checkpoint)
            gpu_ids = [int(x) for x in gpu_ids]
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        else:
            # 如果是在一个卡上训练，多个卡上加载，那么
            gpu_ids = [int(gpu_ids[0])]
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    elif load_type == 'many2one':
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)

            gpu_ids = [int(gpu_ids[0])]
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)
            model.load_state_dict(checkpoint)
        else:
            gpu_ids = [int(x) for x in gpu_ids]

            model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    else:
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            gpu_ids = [int(x) for x in gpu_ids]

            model = torch.nn.DataParallel(model, device_ids=gpu_ids)
            # set_trace()
            model.load_state_dict(checkpoint)
        else:
            gpu_ids = [int(x) for x in gpu_ids]
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    model.to(device)

    return model, device