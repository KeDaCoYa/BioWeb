# -*- encoding: utf-8 -*-
"""
@File    :   normalize_model.py   
@Contact :   1329818994@qq.com
@License :   (C)Copyright 2022-2099
@Author  :   Kedaxia
@Version :   0.0.1
@Create time :   2022/4/6 13:52   
@Description :   None 

"""

import logging
from collections import defaultdict

from ipdb import set_trace

from src.norm.config import MyBertConfig
from src.norm.norm_predicate import init_norm_model
from src.norm.utils.preprocess_utils import TextPreprocess
from src.norm.utils.train import load_cache_dictionary

logger = logging.getLogger("main.norm.norm_model")
class MyNormModel:
    def __init__(self,ckpt_path,config:MyBertConfig):

        self.config = config
        logger.info("开始加载Normalize模型:{}".format(ckpt_path))
        self.model,self.device = init_norm_model(config,ckpt_path)
        logger.info('加载完成Normalize模型....')
        logger.info("开始加载disease信息.....")
        self.dise_dict,self.dise_sparse_embds,self.dise_dense_embds = load_cache_dictionary(config.disease_cache_path)
        logger.info("开始加载chem_drug词典信息.....")
        self.chem_drug_dict,self.chem_drug_sparse_embds,self.chem_drug_dense_embds = load_cache_dictionary(config.chemical_chemical_drug_cache_path)
        logger.info("开始加载gene词典信息.....")
        self.gene_dict,self.gene_sparse_embds,self.gene_dense_embds = load_cache_dictionary(config.gene_protein_cache_path)
        logger.info("开始加载cell_type词典信息.....")
        self.cell_type_dict,self.cell_type_sparse_embds,self.cell_type_dense_embds = load_cache_dictionary(config.cell_type_cache_path)
        logger.info("开始加载chem_line词典信息.....")
        self.cell_line_dict,self.cell_line_sparse_embds,self.cell_line_dense_embds = load_cache_dictionary(config.cell_line_cache_path)
        logger.info("开始加载species词典信息.....")
        self.species_dict,self.species_sparse_embds,self.species_dense_embds = load_cache_dictionary(config.species_cache_path)

    def biosyn_model_predicate(self,input_mention, biosyn, dictionary, dict_sparse_embeds, dict_dense_embeds):
        # preprocess 输入的 mention
        mention = TextPreprocess().run(input_mention)

        # embed mention
        mention_sparse_embeds = biosyn.get_sparse_representation(mentions=[mention])
        mention_dense_embeds = biosyn.get_dense_representation(mentions=[mention])

        output = {
            'mention': input_mention,
        }

        if self.config.verbose:
            output = {
                'mention': input_mention,
                'mention_sparse_embeds': mention_sparse_embeds.squeeze(0),
                'mention_dense_embeds': mention_dense_embeds.squeeze(0)
            }


        # 计算得到sparse score和dense score
        sparse_score_matrix = biosyn.get_score_matrix(
            query_embeds=mention_sparse_embeds,
            dict_embeds=dict_sparse_embeds
        )
        dense_score_matrix = biosyn.get_score_matrix(
            query_embeds=mention_dense_embeds,
            dict_embeds=dict_dense_embeds
        )

        sparse_weight = biosyn.get_sparse_weight().item()

        hybrid_score_matrix = sparse_weight * sparse_score_matrix + dense_score_matrix
        # 获得topk个最相似的单词
        hybrid_candidate_idxs = biosyn.retrieve_candidate(
            score_matrix=hybrid_score_matrix,
            topk=5
        )

        # 只能从字典中获得具体的名称
        predictions = dictionary[hybrid_candidate_idxs].squeeze(0)
        output['predictions'] = []

        for prediction in predictions:
            predicted_name = prediction[0]
            predicted_id = prediction[1]
            output['predictions'].append({
                'name': predicted_name,
                'id': predicted_id
            })

        return output['predictions']

    def entity_normalization(self,entities):
        """
        直接对
        """

        new_entities = []

        for idx, ent in enumerate(entities):
            counter = defaultdict(int)
            ent_name = ent['entity_name']
            ent_type = ent['entity_type']
            ent_id = ent['id']

            if ent_type == 'Disease':
                synonyms = self.biosyn_model_predicate(ent_name, self.model, self.dise_dict, self.dise_sparse_embds,self.dise_dense_embds)

            elif ent_type == 'Gene/Protein' or ent_type=='DNA' or ent_type=='RNA':

                synonyms = self.biosyn_model_predicate(ent_name, self.model, self.gene_dict,
                                                            self.gene_sparse_embds, self.gene_dense_embds)
            elif ent_type == 'cell_line':
                synonyms = self.biosyn_model_predicate(ent_name, self.model, self.cell_line_dict,
                                                            self.cell_line_sparse_embds, self.cell_line_dense_embds)
            elif ent_type == 'cell_type':
                synonyms = self.biosyn_model_predicate(ent_name, self.model, self.cell_type_dict,
                                                            self.cell_type_sparse_embds, self.cell_type_dense_embds)
            elif ent_type == 'Species':

                synonyms = self.biosyn_model_predicate(ent_name, self.model, self.species_dict,
                                                             self.species_sparse_embds, self.species_dense_embds)
            for syn in synonyms:
                counter[syn['id']] += 1

            sort_ent = sorted(counter.items(),key=lambda x:x[1],reverse=True)
            if sort_ent[0][1] > 1:
                most_prob_id = sort_ent[0][0]
                most_norm_name = ''
                for s in synonyms:
                    if s['id'] == most_prob_id:
                        most_norm_name = s['name']
                        break
            else:
                most_prob_id = synonyms[0]['id']
                most_norm_name = synonyms[0]['name']

            ent['norm_id'] = most_prob_id
            ent['norm_name'] = most_norm_name
            new_entities.append(ent)
        return new_entities

