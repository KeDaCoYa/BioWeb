# -*- encoding: utf-8 -*-
"""
@File    :   config.py   
@Contact :   1329818994@qq.com
@License :   (C)Copyright 2022-2099
@Author  :   Kedaxia
@Version :   0.0.1
@Create time :   2022/4/6 8:45   
@Description :   None 

"""

from transformers.configuration_bert import BertConfig


class MyBertConfig(BertConfig):

    def __init__(self, **kwargs):
        '''
        这里Config必须得基于BertConfig，不然一些地方用不了....
        :param kwargs:
        '''
        super(MyBertConfig, self).__init__(**kwargs)


        self.bert_name = kwargs['bert_name']
        self.bert_dir = kwargs['bert_dir']

        self.batch_size = kwargs['batch_size']
        self.max_len = kwargs['max_len']  # 这里一般设置为25
        self.seed = kwargs['seed']

        self.dropout_prob = 0.1

        self.use_gpu = kwargs['use_gpu']
        self.gpu_id = kwargs['gpu_id']

        self.logfile_name = kwargs['logfile_name']


        # 已经计算完成的字典
        self.disease_dictionary_path = "./src/norm/dictionary/disease_dictionary.txt"
        self.disease_cache_path = "./src/norm/cache/cache_disease.pk"

        self.chemical_chemical_drug_dictionary_path = "./src/norm/dictionary/chemical_and_drug_dictionary.txt"
        self.chemical_chemical_drug_cache_path = "./src/norm/cache/cache_chemical_drug.pk"

        self.cell_type_dictionary_path = "./src/norm/dictionary/cell_type_dictionary.txt"
        self.cell_type_cache_path = "./src/norm/cache/cache_cell_type.pk"

        self.cell_line_dictionary_path = "./src/norm/dictionary/cell_line_dictionary.txt"
        self.cell_line_cache_path = "./src/norm/cache/cache_cell_line.pk"

        self.gene_protein_dictionary_path = "./src/norm/dictionary/entre_gene_dictionary.txt"
        self.gene_protein_cache_path = "./src/norm/cache/cache_entre_gene.pk"

        self.species_dictionary_path = "./src/norm/dictionary/species_dictionary.txt"
        self.species_cache_path = "./src/norm/cache/cache_species.pk"


        self.topk = 5


