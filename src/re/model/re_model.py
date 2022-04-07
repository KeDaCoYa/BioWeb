# -*- encoding: utf-8 -*-
"""
@File    :   re_model.py   
@Contact :   1329818994@qq.com
@License :   (C)Copyright 2022-2099
@Author  :   Kedaxia
@Version :   0.0.1
@Create time :   2022/4/5 14:31   
@Description :   None 

"""

import logging
import pickle
import time
from collections import Counter

from ipdb import set_trace
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from src.re.dataset_utils.mtb_dataset import MTBDataset
from src.re.dataset_utils.normal_dataset import NormalDataset
from src.re.dataset_utils.preprocess_utils import transform_normal_data
from src.re.re_predicate import init_re_model
from src.re.utils.train_utils import set_tokenize_special_tag, relation_classification_decode

logger = logging.getLogger("main.re.re_model")

class MyReModel:
    def __init__(self,ckpt_path,config):
        self.config = config
        self.model,self.tokenizer,self.device = init_re_model(ckpt_path)

    def relation_extraction(self,all_dataset):
        """
        :param : all_dataset就是所有需要关系分类的数据
        """
        # 读取所有的相关数据

        if isinstance(all_dataset[0],list):
            examples = transform_normal_data(all_dataset)
        else:
            examples = all_dataset

        set_tokenize_special_tag(self.config, self.tokenizer)
        set_tokenize_special_tag(self.config, self.tokenizer)
        # 这个针对sentence-level的关系分类
        if self.config.data_format == 'normal':
            dev_dataset = NormalDataset(examples, config=self.config, label2id=None, tokenizer=self.tokenizer, device=self.device)
        # MTB的方法，这个至少可以解决一些cross-sentence 关系分类
        elif self.config.data_format == 'mtb':
            dev_dataset = MTBDataset(examples, config=self.config, label2id=None, tokenizer=self.tokenizer, device=self.device)
        else:
            raise ValueError

        dev_dataloader = DataLoader(dataset=dev_dataset, shuffle=False, collate_fn=dev_dataset.collate_fn_predicate,
                                    num_workers=0, batch_size=self.config.batch_size)


        all_predicate_tokens = []
        relation_predicate_res = []
        self.model.eval()
        with torch.no_grad():
            for step, batch_data in tqdm(enumerate(dev_dataloader), desc='正在预测数据...', total=len(dev_dataloader)):
                start_time = time.time()
                if self.config.data_format == 'mtb':
                    input_ids, token_type_ids, attention_masks, e1_mask, e2_mask = batch_data
                    labels = None
                    logits = self.model(input_ids, token_type_ids, attention_masks, labels, e1_mask, e2_mask)
                    predicate_token = relation_classification_decode(logits)

                elif self.config.data_format == 'normal':
                    input_ids, token_type_ids, attention_masks, e1_mask, e2_mask = batch_data
                    labels = None
                    input_ids.to(self.device)
                    token_type_ids.to(self.device)
                    attention_masks.to(self.device)
                    e1_mask.to(self.device)
                    e2_mask.to(self.device)
                    logits = self.model(input_ids, token_type_ids, attention_masks, labels, e1_mask, e2_mask)
                    predicate_token = relation_classification_decode(logits)

                else:
                    raise ValueError

                # 保证他们实体之间存在interaction

                id2label = {
                    0: 'None',
                    1: 'PPI',
                    2: 'DDI',
                    3: 'CPI',
                    4: 'GDI',
                    5: 'CDI',
                }

                for idx in range(self.config.batch_size):
                    try:
                        flag = predicate_token[idx]
                    except:
                        break
                    if flag:
                        if self.config.num_labels == 6:
                            relation_predicate_res.append({
                                'id': 'r' + str(step * self.config.batch_size + idx),
                                'abstract_id': examples[step * self.config.batch_size + idx].abstract_id,
                                'e1_id': examples[step * self.config.batch_size + idx].ent1_id,
                                'e2_id': examples[step] * self.config.batch_size + idx.ent2_id,
                                'relation_type': id2label[predicate_token],
                            })

                        elif self.config.num_labels == 2:

                            relation_predicate_res.append({
                                'id': 'r' + str(step * self.config.batch_size + idx),
                                'abstract_id': examples[step * self.config.batch_size + idx].abstract_id,
                                'e1_id': examples[step * self.config.batch_size + idx].ent1_id,
                                'e2_id': examples[step * self.config.batch_size + idx].ent2_id,
                                'relation_type': 1,
                            })



        return relation_predicate_res
