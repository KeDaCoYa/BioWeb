# -*- encoding: utf-8 -*-
"""
@File    :   ner_model.py   
@Contact :   1329818994@qq.com
@License :   (C)Copyright 2022-2099
@Author  :   Kedaxia
@Version :   0.0.1
@Create time :   2022/4/5 13:02   
@Description :   None 

"""
import logging
from collections import defaultdict
from copy import deepcopy
from itertools import combinations

import torch
from tqdm import tqdm

from src.ner.ner_predicate import init_ner_model
from src.ner.utils.model_utils import span_predicate
from src.ner.utils.preprocess_utils import preprocess_input

logger = logging.getLogger("main.ner.ner_model")
class MyNERModel:
    def __init__(self,ckpt_path,config):
        self.config = config
        self.model,self.tokenizer,self.device = init_ner_model(ckpt_path)

    def entity_extraction(self,text):
        """
            这里给定文本，输出所有的实体
        """
        dev_dataset, dev_loader = preprocess_input(text,self.config,self.tokenizer)
        all_entities = []

        ent_id = 0

        self.model.eval()
        with torch.no_grad():
            for step, batch_data in enumerate(dev_loader):
                entities = []
                if self.config.which_model == 'bert':

                    if self.config.model_name == 'bert_span':
                        raw_text_list, token_ids, attention_masks, token_type_ids, origin_to_subword_index, input_true_length, entity_type_ids = batch_data

                        token_ids, attention_masks, token_type_ids = token_ids.to(self.device), attention_masks.to(
                            self.device), token_type_ids.to(self.device)
                        input_true_length = input_true_length.to(self.device)
                        entity_type_ids = entity_type_ids.to(self.device)

                        tmp_start_logits, tmp_end_logits = self.model(token_ids, attention_masks=attention_masks,
                                                                 token_type_ids=token_type_ids,
                                                                 input_token_starts=origin_to_subword_index,
                                                                 start_ids=None,
                                                                 end_ids=None,
                                                                 input_true_length=input_true_length,
                                                                 entity_type_ids=entity_type_ids)

                        dise_start_logits, chem_start_logits, gene_start_logits, spec_start_logits, celltype_start_logits, cellline_start_logits, dna_start_logits, rna_start_logits = tmp_start_logits
                        dise_end_logits, chem_end_logits, gene_end_logits, spec_end_logits, celltype_end_logits, cellline_end_logits, dna_end_logits, rna_end_logits = tmp_end_logits

                        _, dise_span_start_logits = torch.max(dise_start_logits, dim=-1)
                        _, dise_span_end_logits = torch.max(dise_end_logits, dim=-1)

                        _, chem_span_start_logits = torch.max(chem_start_logits, dim=-1)
                        _, chem_span_end_logits = torch.max(chem_end_logits, dim=-1)

                        _, gene_span_start_logits = torch.max(gene_start_logits, dim=-1)
                        _, gene_span_end_logits = torch.max(gene_end_logits, dim=-1)

                        _, spec_span_start_logits = torch.max(spec_start_logits, dim=-1)
                        _, spec_span_end_logits = torch.max(spec_end_logits, dim=-1)

                        _, celltype_span_start_logits = torch.max(celltype_start_logits, dim=-1)
                        _, celltype_span_end_logits = torch.max(celltype_start_logits, dim=-1)

                        _, cellline_span_start_logits = torch.max(cellline_start_logits, dim=-1)
                        _, cellline_span_end_logits = torch.max(cellline_end_logits, dim=-1)

                        _, dna_span_start_logits = torch.max(dna_start_logits, dim=-1)
                        _, dna_span_end_logits = torch.max(dna_end_logits, dim=-1)

                        _, rna_span_start_logits = torch.max(rna_start_logits, dim=-1)
                        _, rna_span_end_logits = torch.max(rna_end_logits, dim=-1)

                        dise_span_start_logits = dise_span_start_logits.cpu().numpy()
                        dise_span_end_logits = dise_span_end_logits.cpu().numpy()

                        chem_span_start_logits = chem_span_start_logits.cpu().numpy()
                        chem_span_end_logits = chem_span_end_logits.cpu().numpy()

                        gene_span_start_logits = gene_span_start_logits.cpu().numpy()
                        gene_span_end_logits = gene_span_end_logits.cpu().numpy()

                        spec_span_start_logits = spec_span_start_logits.cpu().numpy()
                        spec_span_end_logits = spec_span_end_logits.cpu().numpy()

                        celltype_span_start_logits = celltype_span_start_logits.cpu().numpy()
                        celltype_span_end_logits = celltype_span_end_logits.cpu().numpy()

                        cellline_span_start_logits = cellline_span_start_logits.cpu().numpy()
                        cellline_span_end_logits = cellline_span_end_logits.cpu().numpy()

                        dna_span_start_logits = dna_span_start_logits.cpu().numpy()
                        dna_span_end_logits = dna_span_end_logits.cpu().numpy()

                        rna_span_start_logits = rna_span_start_logits.cpu().numpy()
                        rna_span_end_logits = rna_span_end_logits.cpu().numpy()

                        dise_entities = span_predicate(dise_span_start_logits, dise_span_end_logits, raw_text_list,
                                                       self.config.span_id2label)
                        chem_entities = span_predicate(chem_span_start_logits, chem_span_end_logits, raw_text_list,
                                                       self.config.span_id2label)
                        gene_entities = span_predicate(gene_span_start_logits, gene_span_end_logits, raw_text_list,
                                                       self.config.span_id2label)
                        spec_entities = span_predicate(spec_span_start_logits, spec_span_end_logits, raw_text_list,
                                                       self.config.span_id2label)
                        celltype_entities = span_predicate(celltype_span_start_logits, celltype_span_end_logits,
                                                           raw_text_list, self.config.span_id2label)
                        cellline_entities = span_predicate(cellline_span_start_logits, cellline_span_end_logits,
                                                           raw_text_list, self.config.span_id2label)
                        dna_entities = span_predicate(dna_span_start_logits, dna_span_end_logits, raw_text_list,
                                                      self.config.span_id2label)
                        rna_entities = span_predicate(rna_span_start_logits, rna_span_end_logits, raw_text_list,
                                                      self.config.span_id2label)
                        entities.extend(dise_entities)
                        entities.extend(chem_entities)
                        entities.extend(gene_entities)
                        entities.extend(spec_entities)
                        entities.extend(celltype_entities)
                        entities.extend(cellline_entities)
                        entities.extend(dna_entities)
                        entities.extend(rna_entities)
                else:
                    raise ValueError('选择normal,bert....')

                if entities.__len__() != 0:
                    # all_entities.append({'raw_text':raw_text_list[0],'entities':entities,'sent_pos':step})
                    new_entities = []
                    for ent in entities:
                        # 实体长度太长给pass
                        if int(ent['end_idx']) - int(ent['start_idx']) > 8:
                            continue
                        ent['pos'] = step
                        ent['id'] = "e"+str(ent_id)
                        ent_id += 1
                        new_entities.append(ent)

                    all_entities.extend(new_entities)

        return all_entities

    def reprots(self,entities):
        """
        对这次抽取的entities进行统计
        """
        logger.info("实体抽取个数为:{}".format(len(entities)))

        counter = defaultdict(int)
        for ent in entities:
            counter[ent['entity_type']] += 1
        logger.info(counter)



