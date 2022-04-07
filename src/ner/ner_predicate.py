# -*- encoding: utf-8 -*-
"""
@File    :   ner_predicate.py   
@Contact :   1329818994@qq.com
@License :   (C)Copyright 2022-2099
@Author  :   Kedaxia
@Version :   0.0.1
@Create time :   2022/4/4 21:03   
@Description :   这个就是模型的预测，输入文本，给出结果
                 固定模型为multi_task bert span

"""
import logging
import os

import torch
from transformers import AutoTokenizer, BertTokenizer

from src.ner.config import MyBertConfig
from src.ner.model.multi_span import MultiBERT_Span
from src.ner.utils.model_utils import load_model


logger = logging.getLogger("main.ner_predicate")

def get_ner_bert_config():
    """
    这里手动初始化在predicate时候需要的各种参数
    """
    config = MyBertConfig(
        bert_dir="../../embedding/scibert_scivocab_uncased",
        bert_name='biobert',
        model_name="bert_span",
        decoder_layer="span",
        entity_type="multi",
        batch_size=1,
        max_len=512,
        which_model='bert',
        use_gpu=True if torch.cuda.is_available() else False,
        seed=1234,
        fixed_batch_length=True,
        logfile_name="test_log",
        subword_weight_mode="first",
        num_span_class=9,
    )
    return config




def init_ner_model(ckpt_path):
    """
    在网站初始化的时候也初始化相应的模型
    """
    logger.info("正在加载NER模型:{}".format(ckpt_path))
    config = get_ner_bert_config()
    model = MultiBERT_Span(config)
    device = torch.device("cuda") if config.use_gpu else torch.device("cpu")
    # 加载最佳模型
    model = load_model(model, ckpt_path=ckpt_path)
    model.to(device)
    try:
        tokenizer = BertTokenizer(os.path.join(config.bert_dir, 'vocab.txt'))
    except:
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(config.bert_dir, 'vocab.txt'))
    logger.info("NER模型加载完成....")
    return model,tokenizer,device
if __name__ == '__main__':
    ckpt_path = "./src/trained_models/ner/bert_span.pt"
    init_ner_model(ckpt_path)


