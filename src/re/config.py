# -*- encoding: utf-8 -*-
"""
@File    :   config.py
@Contact :   1329818994@qq.com
@License :   (C)Copyright 2022-2099
@Author  :   Kedaxia
@Version :   0.0.1
@Create time :   2022/4/5 14:27
@Description :   None

"""

from transformers import BertConfig


class BaseConfig:
    def __init__(self, **kwargs):
        '''

        :param gpu_ids:
        :param task_type:
        :param ner_type:
        :param model_name: 这个自由写名字，没啥区别
        '''

        self.model_name = kwargs['model_name']
        self.seed = kwargs['seed']

        self.use_gpu = kwargs['use_gpu']
        self.gpu_id = kwargs['gpu_id']
        self.num_labels = 2
        self.fixed_batch_length = True  # 这个参数控制batch的长度是否固定

        self.logfile_name = kwargs['logfile_name']
        self.logs_dir = './outputs/logs/{}/'.format(self.model_name)

        # 最后模型对文本预测的结果存放点
        self.predicate_dir = './outputs/predicate_outputs/{}/'.format(self.model_name)

        self.class_type = kwargs['class_type']



class MyBertConfig(BaseConfig):
    def __init__(self, **kwargs):
        '''
        开始使用argparse控制config的参数
        :param gpu_ids:
        :param task_type
        '''
        BaseConfig.__init__(self, **kwargs)

        self.bert_name = kwargs['bert_name']
        self.bert_dir = kwargs['bert_dir']
        self.batch_size = kwargs['batch_size']
        self.max_len = 512
        self.dropout_prob = 0.1


        self.subword_weight_mode = kwargs['subword_weight_mode']
        # 实体的

        self.scheme = kwargs['scheme']
        self.data_format = kwargs['data_format']

        self.ent1_start_tag = '[s1]'
        self.ent1_end_tag = '[e1]'
        self.ent2_start_tag = '[s2]'
        self.ent2_end_tag = '[e2]'
        self.special_tags = [self.ent1_start_tag, self.ent1_end_tag, self.ent2_start_tag, self.ent2_end_tag]
        self.total_special_toks = 3


class KebioConfig(BertConfig):
    """Configuration for `KebioModel`."""

    def __init__(self,
                 vocab_size,
                 num_entities,
                 max_mentions=15,
                 max_candidate_entities=100,
                 hidden_size=768,
                 entity_size=50,
                 num_hidden_layers=12,
                 num_context_layers=8,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02, **kwargs):
        super(KebioConfig, self).__init__(vocab_size=vocab_size,
                                          hidden_size=hidden_size,
                                          num_hidden_layers=num_hidden_layers,
                                          num_attention_heads=num_attention_heads,
                                          intermediate_size=intermediate_size,
                                          hidden_act=hidden_act,
                                          hidden_dropout_prob=hidden_dropout_prob,
                                          attention_probs_dropout_prob=attention_probs_dropout_prob,
                                          max_position_embeddings=max_position_embeddings,
                                          type_vocab_size=type_vocab_size,
                                          initializer_range=initializer_range, **kwargs)
        self.num_context_layers = num_context_layers
        self.entity_size = entity_size
        self.num_entities = num_entities
        self.max_mentions = max_mentions
        self.max_candidate_entities = max_candidate_entities


class MyKebioConfig(BaseConfig, BertConfig):
    """Configuration for `KebioModel`."""

    def __init__(self,
                 vocab_size=28895,
                 num_entities=477039,
                 max_mentions=50,
                 max_candidate_entities=100,
                 hidden_size=768,
                 entity_size=100,
                 num_hidden_layers=12,
                 num_context_layers=8,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02, **kwargs):
        BaseConfig.__init__(self, **kwargs)
        BertConfig.__init__(self, vocab_size=vocab_size,
                            hidden_size=hidden_size,
                            num_hidden_layers=num_hidden_layers,
                            num_attention_heads=num_attention_heads,
                            intermediate_size=intermediate_size,
                            hidden_act=hidden_act,
                            hidden_dropout_prob=hidden_dropout_prob,
                            attention_probs_dropout_prob=attention_probs_dropout_prob,
                            max_position_embeddings=max_position_embeddings,
                            type_vocab_size=type_vocab_size,
                            initializer_range=initializer_range, **kwargs)

        self.num_context_layers = num_context_layers
        self.entity_size = entity_size
        self.num_entities = num_entities
        self.max_mentions = max_mentions
        self.max_candidate_entities = max_candidate_entities

        self.class_name = 'KeBioLM Config'

        self.bert_name = kwargs['bert_name']
        self.run_type = kwargs['run_type']
        self.bert_dir = kwargs['bert_dir']

        self.batch_size = kwargs['batch_size']


        self.dropout_prob = 0.1


        self.subword_weight_mode = kwargs['subword_weight_mode']
        # 实体的

        self.scheme = kwargs['scheme']
        self.data_format = kwargs['data_format']

        self.ent1_start_tag = '[s1]'
        self.ent1_end_tag = '[e1]'
        self.ent2_start_tag = '[s2]'
        self.ent2_end_tag = '[e2]'
        self.special_tags = [self.ent1_start_tag, self.ent1_end_tag, self.ent2_start_tag, self.ent2_end_tag]
        self.total_special_toks = 3
