# -*- encoding: utf-8 -*-
"""
@File    :   re_predicate.py   
@Contact :   1329818994@qq.com
@License :   (C)Copyright 2022-2099
@Author  :   Kedaxia
@Version :   0.0.1
@Create time :   2022/4/5 14:27   
@Description :   None 

"""

# -*- encoding: utf-8 -*-
from src.re.utils.train_utils import choose_re_model

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

from src.re.config import MyBertConfig

from src.ner.utils.model_utils import load_model


logger = logging.getLogger("main.ner_predicate")

def get_re_bert_config():
    """
    这里手动初始化在predicate时候需要的各种参数
    """
    config = MyBertConfig(
        bert_dir="../../embedding/biobert-base-cased-v1.2",
        bert_name='biobert',
        scheme=1,
        data_format='normal',
        model_name='r_bert',
        seed=1234,
        batch_size=128,
        subword_weight_mode='first',
        gpu_id=0,
        use_gpu=True if torch.cuda.is_available() else False,
        class_type='single',
        logfile_name='log_name'
    )
    return config


def init_re_model(ckpt_path):
    """
    在网站初始化的时候也初始化相应的模型
    """
    logger.info("正在加载RE模型:{}".format(ckpt_path))
    config = get_re_bert_config()
    model = choose_re_model(config)
    device = torch.device("cuda") if config.use_gpu and torch.cuda.is_available() else torch.device("cpu")
    # 加载最佳模型
    model = load_model(model, ckpt_path=ckpt_path)
    model.to(device)
    try:
        tokenizer = BertTokenizer(os.path.join(config.bert_dir, 'vocab.txt'))
    except:
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(config.bert_dir, 'vocab.txt'))
    logger.info("RE模型加载完成....")
    return model,tokenizer,device
if __name__ == '__main__':
    ckpt_path = "./src/trained_models/re/single/rbert/model.pt"
    init_re_model(ckpt_path)


