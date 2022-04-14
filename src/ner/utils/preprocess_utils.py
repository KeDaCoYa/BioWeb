# -*- encoding: utf-8 -*-
"""
@File    :   preprocess_utils.py   
@Contact :   1329818994@qq.com
@License :   (C)Copyright 2022-2099
@Author  :   Kedaxia
@Version :   0.0.1
@Create time :   2022/4/5 11:46   
@Description :   这个文件里面主要是对数据的预处理

"""
from copy import deepcopy
from itertools import combinations

import requests
from ipdb import set_trace
from nltk import wordpunct_tokenize
from nltk.tokenize import sent_tokenize

import random
import time
import json
import requests


from torch.utils.data import DataLoader

from src.ner.dataset_utils.base_dataset import NERProcessor
from src.ner.dataset_utils.bert_span_dataset import BertSpanDataset_dynamic

USER_AGENTS = [
    'Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_6_8; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50',
    'Mozilla/5.0 (Linux;u;Android 4.2.2;zh-cn;) AppleWebKit/534.46 (KHTML,likeGecko) Version/5.1 Mobile Safari/10600.6.3 (compatible; Baiduspider/2.0;+http://www.baidu.com/search/spider.html)',
    'Mozilla/5.0 (iPhone; U; CPU iPhone OS 4_3_3 like Mac OS X; en-us) AppleWebKit/533.17.9 (KHTML, like Gecko) Version/5.0.2 Mobile/8J2 Safari/6533.18.5',
    'Mozilla/5.0 (iPod; U; CPU iPhone OS 4_3_3 like Mac OS X; en-us) AppleWebKit/533.17.9 (KHTML, like Gecko) Version/5.0.2 Mobile/8J2 Safari/6533.18.5',
    'Mozilla/5.0 (iPad; U; CPU OS 4_3_3 like Mac OS X; en-us) AppleWebKit/533.17.9 (KHTML, like Gecko) Version/5.0.2 Mobile/8J2 Safari/6533.18.5',
    'Mozilla/5.0 (Linux; U; Android 2.3.7; en-us; Nexus One Build/FRF91) AppleWebKit/533.1 (KHTML, like Gecko) Version/4.0 Mobile Safari/533.1',
    'MQQBrowser/26 Mozilla/5.0 (Linux; U; Android 2.3.7; zh-cn; MB200 Build/GRJ22; CyanogenMod-7) AppleWebKit/533.1 (KHTML, like Gecko) Version/4.0 Mobile Safari/533.1',
    'Opera/9.80 (Android 2.3.4; Linux; Opera Mobi/build-1107180945; U; en-GB) Presto/2.8.149 Version/11.10',
    'Mozilla/5.0 (Linux; U; Android 3.0; en-us; Xoom Build/HRI39) AppleWebKit/534.13 (KHTML, like Gecko) Version/4.0 Safari/534.13',
    'Mozilla/5.0 (BlackBerry; U; BlackBerry 9800; en) AppleWebKit/534.1+ (KHTML, like Gecko) Version/6.0.0.337 Mobile Safari/534.1+',
    'Mozilla/5.0 (hp-tablet; Linux; hpwOS/3.0.0; U; en-US) AppleWebKit/534.6 (KHTML, like Gecko) wOSBrowser/233.70 Safari/534.6 TouchPad/1.0',
    'Mozilla/5.0 (SymbianOS/9.4; Series60/5.0 NokiaN97-1/20.0.019; Profile/MIDP-2.1 Configuration/CLDC-1.1) AppleWebKit/525 (KHTML, like Gecko) BrowserNG/7.1.18124',
    'Mozilla/5.0 (compatible; MSIE 9.0; Windows Phone OS 7.5; Trident/5.0; IEMobile/9.0; HTC; Titan)',
    "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Win64; x64; Trident/5.0; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 2.0.50727; Media Center PC 6.0)",
    "Mozilla/5.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0; WOW64; Trident/4.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 1.0.3705; .NET CLR 1.1.4322)",
    "Mozilla/4.0 (compatible; MSIE 7.0b; Windows NT 5.2; .NET CLR 1.1.4322; .NET CLR 2.0.50727; InfoPath.2; .NET CLR 3.0.04506.30)",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN) AppleWebKit/523.15 (KHTML, like Gecko, Safari/419.3) Arora/0.3 (Change: 287 c9dfb30)",
]


def preprocess_input(text,config,tokenizer):
    """
    :parm : text就是输入的文本或者abstracts
    """
    sents = sent_tokenize(text)
    sent_li = []
    for sent in sents:
        sent_li.append(wordpunct_tokenize(sent))
    processor = NERProcessor(0)
    train_examples = processor.get_examples(sent_li, None)
    train_dataset = BertSpanDataset_dynamic(config, train_examples, tokenizer)
    train_loader = DataLoader(dataset=train_dataset, shuffle=False, num_workers=0,
                              batch_size=config.batch_size,
                              collate_fn=train_dataset.multi_collate_fn_predicate)
    return train_dataset, train_loader

def get_text_data_from_pubmed(pmid):
    """
    这里使用BERN2的结果来用...
    """
    user_agents = random.choice(USER_AGENTS)
    headers = {}
    headers.update({'user-agent': user_agents})
    headers.update({'accept': 'application/json'})
    headers.update({'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8'})
    headers.update({'accept-encoding': 'gzip, deflate, br'})
    url = 'http://bern2.korea.ac.kr/pubmed/{}'.format(pmid)
    response = requests.get(url,headers=headers)
    status_code = response.status_code

    if status_code == 200:

        res = json.loads(response.text)
        text = res[0]['text']

    else:
        print(f"warn! response.status_code={response.status_code}")
        text = ""

    return text

def generate_entity_pairs(entities):
    # 这里需要限制一下两个实体的类别和实体名称
    li = list(combinations(entities, 2))
    new_li = []
    for ent1, ent2 in li:
        ent1_name = ent1['entity_name']
        ent2_name = ent2['entity_name']
        if ent1_name == ent2_name:
            continue
        ent1_type = ent1['entity_type']
        ent2_type = ent2['entity_type']
        if (ent1_type, ent2_type) in [('Gene/Protein','Gene/Protein'),('Gene/Protein','Chemical/Drug'),
                                      ("Chemical/Drug","Gene/Protein"),("Chemical/Drug","Chemical/Drug"),("Gene/Protein","Disease"),
                                      ("Disease","Gene/Protein"),("Chemical/Drug","Disease"),
                                      ("Disease","Chemical/Drug"),('DNA','Gene/Protein'),('Gene/Protein','DNA')]:
            new_li.append((ent1, ent2))

    return new_li


def generate_re_mtb(abstract_id,sent_li,entities):
    dataset = [['abstract_id', 'sent1', 'sent2', 'ent1_name', 'ent2_name', 'ent1_type', 'ent2_type', 'ent1_id', 'ent2_id',
         'distance']]

    entity_pairs = generate_entity_pairs(entities)

    for ent1, ent2 in entity_pairs:
        ent1_name = ent1['entity_name']
        ent1_start_idx = int(ent1['start_idx'])
        ent1_end_idx = int(ent1['end_idx'])
        ent1_type = ent1['entity_type']
        ent1_pos = int(ent1['pos'])
        ent1_id = (ent1['id'])

        ent2_name = ent2['entity_name']
        ent2_start_idx = int(ent2['start_idx'])
        ent2_end_idx = int(ent2['end_idx'])
        ent2_type = ent2['entity_type']
        ent2_pos = int(ent2['pos'])
        ent2_id = (ent2['id'])

        distance = ent2_pos - ent1_pos
        if distance > 3:
            continue
        sent1 = deepcopy(sent_li[ent1_pos])
        sent2 = deepcopy(sent_li[ent2_pos])

        # 构建sent1格式
        sent1.insert(ent1_start_idx, '[s1]')
        sent1.insert(ent1_end_idx + 2, '[e1]')

        sent2.insert(ent2_start_idx, '[s2]')
        sent2.insert(ent2_end_idx + 2, '[e2]')

        dataset.append([abstract_id, " ".join(sent1), " ".join(sent2), ent1_name, ent2_name, ent1_type, ent2_type,
             ent1_id, ent2_id, str(distance)])

    return dataset
def generate_re_normal(abstract_id,sent_li,entities):
    dataset = [['abstract_id', 'sent', 'ent1_name', 'ent2_name', 'ent1_type', 'ent2_type', 'ent1_id', 'ent2_id',
         'distance']]
    max_length = 560
    entity_pairs = generate_entity_pairs(entities)
    for ent1, ent2 in entity_pairs:
        ent1_name = ent1['entity_name']
        ent1_start_idx = int(ent1['start_idx'])
        ent1_end_idx = int(ent1['end_idx'])
        ent1_type = ent1['entity_type']
        ent1_pos = int(ent1['pos'])
        ent1_id = ent1['id']

        ent2_name = ent2['entity_name']
        ent2_start_idx = int(ent2['start_idx'])
        ent2_end_idx = int(ent2['end_idx'])
        ent2_type = ent2['entity_type']
        ent2_pos = int(ent2['pos'])
        ent2_id = ent2['id']

        distance = ent2_pos - ent1_pos
        if distance > 3:
            continue
        sent1 = deepcopy(sent_li[ent1_pos])
        sent2 = deepcopy(sent_li[ent2_pos])

        all_sent = []
        lens = sum([len(x) for x in sent_li[ent1_pos:ent2_pos + 1]])
        if lens > max_length:
            continue
        # 构建normal格式
        if ent1_pos == ent2_pos:  # 如果两个实体再同一个句子之中

            sent1.insert(ent1_start_idx, '[s1]')
            sent1.insert(ent1_end_idx + 2, '[e1]')

            sent1.insert(ent2_start_idx + 2, '[s2]')
            sent1.insert(ent2_end_idx + 4, '[e2]')
            sent1 = ' '.join(sent1)
            dataset.append([abstract_id, sent1, ent1_name, ent2_name, ent1_type, ent2_type, ent1_id, ent2_id,
                 str(distance)])
        else:
            sent1.insert(ent1_start_idx, '[s1]')
            sent1.insert(ent1_end_idx + 2, '[e1]')

            sent2.insert(ent2_start_idx, '[s2]')
            sent2.insert(ent2_end_idx + 2, '[e2]')

            all_sent = []
            all_sent.extend(sent1)
            for sent in sent_li[ent1_pos + 1:ent2_pos]:
                all_sent.extend(sent)
            all_sent.extend(sent2)
            all_sent = ' '.join(all_sent)
            dataset.append([abstract_id, all_sent, ent1_name, ent2_name, ent1_type, ent2_type, ent1_id, ent2_id,
                 str(distance)])

    print("合成数据集的个数为:{}".format(len(dataset)))
    return dataset