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
import torch
from ipdb import set_trace

from config import BertConfig
from src.models.bert_model import BaseBert
import torch.nn as nn

class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.0, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)


class RBERT(BaseBert):
    def __init__(self, config:BertConfig):
        super(RBERT, self).__init__(config)

        self.num_labels = config.num_labels
        self.config = config
        # 下面这两个dim可以进行修改
        self.cls_dim = self.bert_config.hidden_size
        self.entity_dim = self.bert_config.hidden_size
        # 这是对cls的线性变换...
        # (cls_fc_layer): FCLayer(
        #     (dropout): Dropout(p=0.1, inplace=False)
        # (linear): Linear(in_features=768, out_features=768, bias=True)
        # (tanh): Tanh()
        # )
        self.cls_fc_layer = FCLayer(self.bert_config.hidden_size, self.cls_dim,self.config.dropout_prob)
        #   (entity_fc_layer): FCLayer(
        #     (dropout): Dropout(p=0.1, inplace=False)
        #     (linear): Linear(in_features=768, out_features=768, bias=True)
        #     (tanh): Tanh()
        #   )
        self.entity_fc_layer = FCLayer(self.bert_config.hidden_size, self.entity_dim, self.config.dropout_prob)
        # (label_classifier): FCLayer(
        #     (dropout): Dropout(p=0.1, inplace=False)
        #     (linear): Linear(in_features=2304, out_features=19, bias=True)
        #     (tanh): Tanh()
        #   )
        self.label_classifier = FCLayer(
            self.cls_dim+self.entity_dim*2,
            self.config.num_labels,
            self.config.dropout_prob,
            use_activation=False,
        )
        if self.config.freeze_bert:
            self.freeze_parameter(config.freeze_layers)

        # 模型层数的初始化初始化
        nn.init.xavier_normal_(self.cls_fc_layer.linear.weight)
        nn.init.constant_(self.cls_fc_layer.linear.bias, 0.)
        nn.init.xavier_normal_(self.entity_fc_layer.linear.weight)
        nn.init.constant_(self.cls_fc_layer.linear.bias, 0.)


    @staticmethod
    def entity_average(hidden_output,entity_mask):
        '''
        根据mask来获得对应的输出
        :param hidden_output:hidden_output是bert的输出，shape=(batch_size,seq_len,hidden_size)=(16,128,756)
         :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, hidden_dim]
        '''
        e_mask_unsqueeze = entity_mask.unsqueeze(1) #shape=(batch_size,1,seq_len)
        # 这相当于获得实体的实际长度
        length_tensor = (entity_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]
        # [batch_size, 1, seq_len] * [batch_size, seq_len, hidden_dim] = [batch_size, 1, hidden_dim] -> [batch_size, hidden_dim]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting

        return avg_vector
    def forward(self, input_ids, token_type_ids,attention_masks, labels,e1_mask, e2_mask):
        '''
        但是这里将实体对放在一个[CLS]sent<SEP>中，而不是两个sent之中

        :param input_ids:
        :param token_type_ids:
        :param attention_masks:
        :param e1_mask:  这里e1_mask和e2_mask覆盖了special tag，所以这里需要需要切片以下
        :param e2_mask:
        :param labels:
        :return:
        '''

        outputs = self.bert_model(
            input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0] # shape=(batch_size,seq_len,hidden_size)
        pooled_output = outputs[1]  # [CLS],shape = (batch_size,hidden_size)=(16,768)

        # Average
        # e1_h.shape=(batch_size,768)
        # if self.config.scheme%2 == 1:
        #     e1_mask_ = e1_mask.cpu().numpy().tolist()
        #     e2_mask_ = e2_mask.cpu().numpy().tolist()
        #     set_trace()
        #     e1_start_idx, e1_end_idx = self.get_ent_position(e1_mask_)
        #     e2_start_idx, e2_end_idx = self.get_ent_position(e2_mask_)
        #     e1_mask[e1_start_idx] = 0
        #     e1_mask[e1_end_idx] = 0
        #     e2_mask[e2_start_idx] = 0
        #     e2_mask[e2_end_idx] = 0
        e1_h = self.entity_average(sequence_output, e1_mask)
        e2_h = self.entity_average(sequence_output, e2_mask)

        # Dropout -> tanh -> fc_layer (Share FC layer for e1 and e2)
        # pooled_output.shape=(batch_size,768)
        pooled_output = self.cls_fc_layer(pooled_output)

        e1_h = self.entity_fc_layer(e1_h)
        e2_h = self.entity_fc_layer(e2_h)


        # Concat -> fc_layer
        concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)#torch.Size([16, 2304])
        logits = self.label_classifier(concat_h)



        # Softmax
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            return loss,logits

        return logits  # (loss), logits, (hidden_states), (attentions)


