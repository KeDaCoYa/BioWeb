# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :  
   Author :        kedaxia
   date：          2021/12/03
   Copyright:      (c) kedaxia 2021
-------------------------------------------------
   Change Activity:
                   2021/12/03: 
-------------------------------------------------
"""

from ipdb import set_trace

import numpy as np

import torch
import torch.nn as nn

from transformers import BertModel
import logging

logger = logging.getLogger('main.bert_model')


class BaseBert(nn.Module):
    def __init__(self, config):
        """
        这是最基础的BERT模型加载，加载预训练的模型
        :param config:
        :param bert_dir:
        :param dropout_prob:
        """
        super(BaseBert, self).__init__()

        self.config = config
        if config.bert_name == 'kebiolm':

            pass
        else:

            self.bert_model = BertModel.from_pretrained(config.bert_dir, output_hidden_states=True,hidden_dropout_prob=config.dropout_prob)

        self.bert_config = self.bert_model.config

    @staticmethod
    def special_tag_representation(seq_output, input_ids, special_tag):
        '''
        这里就是根据special_tag来获取对应的representation
        input_ids就是为了定位位置
        '''
        # nonzero是得到坐标，表示在(input_ids == special_tag)中，值不为0的坐标
        spec_idx = (input_ids == special_tag).nonzero(as_tuple=False)

        temp = []
        for idx in spec_idx:
            temp.append(seq_output[idx[0], idx[1], :])
        tags_rep = torch.stack(temp, dim=0)

        return tags_rep

    def get_ent_position(self, e_mask):
        '''
        获得entity mask的start_index和end_index
        :param e_mask: shape=(bs,seq_len)
        :return:
        '''
        start_idx = e_mask.index(1)
        for i in range(start_idx + 1, len(e_mask)):
            if e_mask[i] == 1 and e_mask[i + 1] == 0:
                return (start_idx, i)
        return start_idx, len(e_mask) - 1

    def get_entity_representation(self, bert_outputs, pooled_output, input_ids, e1_mask=None, e2_mask=None):
        '''
        这里使用两个的bert outputs输出...
        :param bert_outputs:
        :param pool_output:
        :param schema:
            这里主要是MTB的方法，这里的四种方式和ClinicalTransformer保持一直，之后可能会增加
        :return: 直接返回最终的new_pooled_output
        '''
        if self.scheme == 1:
            seq_tags = []  # 论文中的(2),使用[CLS]和实体的start 标记(也就是<e1>,<e2>或者说<s1><s2>)
            for each_tag in [self.config.ent1_start_tag_id, self.config.ent2_start_tag_id]:
                seq_tags.append(self.special_tag_representation(bert_outputs, input_ids, each_tag))
            new_pooled_output = torch.cat((pooled_output, *seq_tags), dim=1)

        elif self.scheme == 2:  # 论文中的(3),使用实体1和实体2的<s1><e1>,<s2><e2>，这个效果在clinicalTransformer论文中效果最好....
            seq_tags = []
            for each_tag in [self.config.ent1_start_tag_id, self.config.ent1_end_tag_id, self.config.ent2_start_tag_id,
                             self.config.ent2_end_tag_id]:
                seq_tags.append(self.special_tag_representation(bert_outputs, input_ids, each_tag))
            new_pooled_output = torch.cat((pooled_output, *seq_tags), dim=1)
        elif self.scheme == 3:  # 论文中的(4),只使用两个实体的开始标志：<s1><s2>...
            seq_tags = []
            for each_tag in [self.config.ent1_start_tag_id, self.config.ent2_start_tag_id]:
                seq_tags.append(self.special_tag_representation(bert_outputs, input_ids, each_tag))
            new_pooled_output = torch.cat(seq_tags, dim=1)
        elif self.scheme == 4:  # 这是论文中介绍的(1),只使用[CLS]的output
            new_pooled_output = pooled_output  # shape=(batch_size,hidden_size*2)
        elif self.scheme == 5:  # 这个是最基本的情况，直接将e1_mask和e2_mask对应的全部拿来

            e1_mask = e1_mask.unsqueeze(1)
            e2_mask = e2_mask.unsqueeze(1)
            ent1_rep = torch.bmm(e1_mask.float(), bert_outputs)
            ent2_rep = torch.bmm(e2_mask.float(), bert_outputs)
            ent1_rep = ent1_rep.squeeze(1)
            ent2_rep = ent2_rep.squeeze(1)
            new_pooled_output = torch.cat([ent1_rep, ent2_rep], dim=1)
        elif self.scheme == 6:  # 只获得真正实体对应的部分，取消掉[s1][e1],[s2][e2]

            # 取消e1_mask,e2_mask在[s1][e1],[s2][e2]的label，也就是直接设为0
            # e1_start_idx, e1_end_idx = self.get_ent_position(e1_mask)
            # e2_start_idx, e2_end_idx = self.get_ent_position(e2_mask)
            # e1_mask[e1_start_idx] = 0
            # e1_mask[e1_end_idx] = 0
            # e2_mask[e2_start_idx] = 0
            # e2_mask[e2_end_idx] = 0
            bs,seq_len = e1_mask.shape
            tmp_e1_mask = e1_mask.cpu().numpy().tolist()
            tmp_e2_mask = e2_mask.cpu().numpy().tolist()
            for i in range(bs):
                tmp_e1 = tmp_e1_mask[i]
                tmp_e2 = tmp_e2_mask[i]
                start_idx_e1 =tmp_e1.index(0)
                end_idx_e1 = start_idx_e1+sum(tmp_e1)-1
                start_idx_e2 =tmp_e2.index(0)
                end_idx_e2 = start_idx_e2 + sum(tmp_e2) - 1
                e1_mask[start_idx_e1][end_idx_e1] = 0
                e2_mask[start_idx_e2][end_idx_e2] = 0

            e1_mask = e1_mask.unsqueeze(1)
            e2_mask = e2_mask.unsqueeze(1)
            ent1_rep = torch.bmm(e1_mask.float(), bert_outputs)
            ent2_rep = torch.bmm(e2_mask.float(), bert_outputs)
            ent1_rep = ent1_rep.squeeze(1)
            ent2_rep = ent2_rep.squeeze(1)
            new_pooled_output = torch.cat([ent1_rep, ent2_rep], dim=1)
        else:
            raise ValueError
        return new_pooled_output
