# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :  
   Author :        kedaxia
   date：          2021/12/22
   Copyright:      (c) kedaxia 2021
-------------------------------------------------
   Change Activity:
                   2021/12/22: 
-------------------------------------------------
"""

from ipdb import set_trace
import numpy as np

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss,BCELoss

from config import BertConfig
from src.models.bert_model import BaseBert


class MTBRelationClassification(BaseBert):
    def __init__(self, config:BertConfig):
        super(MTBRelationClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.dropout = nn.Dropout(config.dropout_prob)


        self.criterion = CrossEntropyLoss(reduction='none')
        # 初始化部分网络参数
        # nn.init.xavier_normal_(self.test1_entity.weight)
        # nn.init.constant_(self.test1_entity.bias, 0.)
        # nn.init.xavier_normal_(self.test2_entity.weight)
        # nn.init.constant_(self.test2_entity.bias, 0.)
        self.scheme = config.scheme
        if self.scheme == 1:
            self.classifier_dim = self.bert_config.hidden_size * 3
        elif self.scheme == 2:
            self.classifier_dim = self.bert_config.hidden_size * 5
        elif self.scheme == 3:
            self.classifier_dim = self.bert_config.hidden_size * 2
        elif self.scheme == 4:
            self.classifier_dim = self.bert_config.hidden_size
        elif self.scheme == 5:
            self.classifier_dim = self.bert_config.hidden_size * 2
        elif self.scheme == 6:
            self.classifier_dim = self.bert_config.hidden_size * 2
        else:
            raise ValueError('scheme没有此:{}'.format(self.scheme))


        self.classifier = nn.Linear(self.classifier_dim, self.num_labels)



    def forward(self, input_ids, token_type_ids,attention_masks,labels,e1_mask,e2_mask):
        '''
        这个应该支持多卡训练
        :param input_ids: (batch_size,seq_len,hidden_size)
        :param token_type_ids:
        :param attention_masks:
        :param labels:
        :param entity_positions:
        :return:
        '''

        bert_outputs = self.bert_model(input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids)  #返回 sequence_output, pooled_output, (hidden_states), (attentions)

        bert_output = bert_outputs[0] # shape=(batch_size,seq_len,hidden_size)
        pooled_outputs = bert_outputs[1] # shape=(batch_size,seq_len,hidden_size)



        #pooled_output.shape = (batch_size,hiddensize*2)

        # 这里提取目标representation...

        pooled_output = self.get_entity_representation(bert_output,pooled_outputs,input_ids,e1_mask=e1_mask,e2_mask=e2_mask)


        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)


        if labels is not None:
            loss = self.criterion(logits.view(-1, self.num_labels), labels.view(-1))
            return loss,logits

        return logits


