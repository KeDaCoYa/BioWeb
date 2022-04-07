# -*- encoding: utf-8 -*-
"""
@File    :   config.py   
@Contact :   1329818994@qq.com
@License :   (C)Copyright 2022-2099
@Author  :   Kedaxia
@Version :   0.0.1
@Create time :   2022/4/4 21:16   
@Description :   None 

"""
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :  这是所有模型的配置文件
   Author :        kedaxia
   date：          2021/11/08
   Copyright:      (c) kedaxia 2021
-------------------------------------------------
   Change Activity:
                   2021/11/08: 今天是个好日期
-------------------------------------------------
"""
import json

from ipdb import set_trace
from transformers import BertConfig


class BaseConfig:
    def __init__(self, **kwargs):
        '''

        :param gpu_ids:
        :param model_name:
        :param ner_type:
        :param model_name: 这个自由写名字，没啥区别
        '''
        # 这个是选择数据集name
        self.which_model = kwargs['which_model']
        self.model_name = kwargs['model_name']
        self.decoder_layer = kwargs['decoder_layer']
        self.seed = kwargs['seed']
        self.entity_type = kwargs['entity_type']

        self.use_gpu = kwargs['use_gpu']

        self.fixed_batch_length = kwargs['fixed_batch_length']  # 这个参数控制batch的长度是否固定

        self.logfile_name = kwargs['logfile_name']
        self.logs_dir = './outputs/logs/{}/'.format(self.model_name)



        self.span_label2id_path = './src/ner/dataset/span_label2id.json'
        self.span_label2id = json.load(open(self.span_label2id_path, 'r', encoding='utf-8'))
        self.span_id2label_path = './src/ner/dataset/span_id2label.json'
        self.span_id2label = json.load(open(self.span_id2label_path, 'r', encoding='utf-8'))
        self.span_id2label = {int(key): value for key, value in self.span_id2label.items()}
        # span指针的classses
        # 这个是表示实体类别的个数，但是要包括非实体O，例如实体类别有疾病、化学、蛋白质，那么类别个数为3+1=4
        self.num_span_class = 9
        self.predicate_flag = True  # 在训练和验证的时候都是False，只有面对无标签的测试集才会打开


class MyBertConfig(BertConfig,BaseConfig):
    def __init__(self, **kwargs):
        '''
        开始使用argparse控制config的参数

        :param model_name:
        :param ner_type:
        '''


        BaseConfig.__init__(self,**kwargs)
        BertConfig.__init__(self,**kwargs)

        self.bert_dir = kwargs['bert_dir']
        self.bert_name = kwargs['bert_name']
        self.batch_size = kwargs['batch_size']
        self.max_len = kwargs['max_len']  # BC5CDR-disease的文本最长为121
        self.subword_weight_mode = kwargs['subword_weight_mode']
        self.dropout_prob = 0.1


class KebioConfig(BertConfig):
    """Configuration for `KebioModel`."""

    def __init__(self, vocab_size, num_entities, **kwargs):
        super(KebioConfig, self).__init__()
        self.vocab_size = vocab_size
        self.num_entities = num_entities

        self.max_mentions = 50
        self.max_candidate_entities = 100
        self.hidden_size = 768

        self.which_model = kwargs['which_model']

        self.entity_size = 100  # 这个应该是entity embedding dim
        self.num_hidden_layers = 12
        self.num_context_layers = 8
        self.num_attention_heads = 12
        self.intermediate_size = 3072
        self.hidden_act = "gelu"
        self.model_type = "bert"
        self.pad_token_id = 0
        self.layer_norm_eps = 1e-12
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.max_position_embeddings = 512

        # 这个是token_type_embedding的token_type个数，很神奇...
        self.type_vocab_size = 2
        self.initializer_range = 0.02

        self.bert_dir = kwargs['bert_dir']
        self.bert_name = kwargs['bert_name']


        self.model_name = kwargs['model_name']
        self.decoder_layer = kwargs['decoder_layer']


        self.batch_size = kwargs['batch_size']

        self.max_len = kwargs['max_len']  # BC5CDR-disease的文本最长为121
        self.entity_type = kwargs['entity_type']  # BC5CDR-disease的文本最长为121


        self.subword_weight_mode = kwargs['subword_weight_mode']


        self.model_name = kwargs['model_name']
        self.seed = kwargs['seed']

        self.use_gpu = kwargs['use_gpu']
        self.fixed_batch_length = kwargs['fixed_batch_length']  # 这个参数控制batch的长度是否固定

        self.logfile_name = kwargs['logfile_name']

        self.span_label2id_path = './src/ner/dataset/span_label2id.json'
        self.span_label2id = json.load(open(self.span_label2id_path, 'r', encoding='utf-8'))
        self.span_id2label_path = './src/ner/dataset/span_id2label.json'
        self.span_id2label = json.load(open(self.span_id2label_path, 'r', encoding='utf-8'))
        self.span_id2label = {int(key): value for key, value in self.span_id2label.items()}
        # span指针的classses
        # 这个是表示实体类别的个数，但是要包括非实体O，例如实体类别有疾病、化学、蛋白质，那么类别个数为3+1=4
        self.num_span_class = 9

        self.predicate_flag = True  # 在训练和验证的时候都是False，只有面对无标签的测试集才会打开

