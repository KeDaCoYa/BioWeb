# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :  数据读取
   Author :        kedaxia
   date：          2021/12/02
   Copyright:      (c) kedaxia 2021
-------------------------------------------------
   Change Activity:
                   2021/12/02: 
-------------------------------------------------
"""




from ipdb import set_trace

import numpy as np



class InputExamples(object):
    def __init__(self,text,label,ent1_type,ent2_type,ent1_name,ent2_name,ent1_id=None,ent2_id=None,abstract_id=None):
        '''
        针对sentence-level的关系分类任务....
        :param text_a:
        :param text_b:
        :param label:
        :param ent1_type:
        :param ent2_type:
        '''
        self.text = text
        self.label = label
        self.ent1_type = ent1_type
        self.ent2_type = ent2_type
        self.ent1_name = ent1_name
        self.ent2_name = ent2_name
        self.ent1_id = ent1_id
        self.ent2_id = ent2_id
        self.abstract_id = abstract_id

class MTBExamples(object):
    def __init__(self,text_a,text_b,label,ent1_type,ent2_type,ent1_name=None,ent2_name=None,ent1_id=None,ent2_id=None,abstract_id=None):
        '''
        MTB的cross-sentence 关系分类任务
        :param text_a:
        :param text_b:
        :param label:
        :param ent1_type:
        :param ent2_type:
        :param ent1_name:
        :param ent2_name:
        '''
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.ent1_type = ent1_type
        self.ent2_type = ent2_type
        self.ent1_name = ent1_name
        self.ent2_name = ent2_name
        self.ent1_id = ent1_id
        self.ent2_id = ent2_id
        self.abstract_id = abstract_id


def get_relative_pos_feature(x,limit):
     """
        :param x = idx - entity_idx
        这个方法就是不管len(sentence)多长，都限制到这个位置范围之内

        x的范围就是[-len(sentence),len(sentence)] 转换到都是正值范围
        -limit ~ limit => 0 ~ limit * 2+2
        将范围转换一下，为啥
    """
     if x < -limit:
         return 0
     elif x >= -limit and x <= limit:
         return x + limit + 1
     else:
         return limit * 2 + 2

def get_label2id(label_file):
    f = open(label_file,'r')
    t = f.readlines()
    f.close()
    label2id = {}
    id2label = {}
    for i,label in enumerate(t):
        label = label.strip()
        label2id[label] = i
        id2label[i] = label

    return label2id,id2label


def read_file(file_path):
    f = open(file_path, 'r', encoding='utf-8')
    t = f.readlines()
    f.close()
    res = [x.strip() for x in t]
    return res

def read_raw_data(config):
    '''
    这里根据不同的数据集，需要读取不同格式的数据集，但是最后输出会保持一致，一个是sentence，另一个是label
    :param config:
    :param type:
    :return:
    '''

    if config.data_format == 'normal': #格式为<CLS>sentence a<sep>sentence b <sep>
        examples = process_raw_normal_data(config.dev_normal_path)
    elif config.data_format == 'mtb':
        examples = process_raw_mtb_data(config.dev_mtb_path)
    else:
       raise ValueError("data_format错误")
    return examples

def transform_mtb_data(dataset):
    res = []
    for idx, line in enumerate(dataset[1:]):
        abstract_id, sent1, sent2, ent1_name, ent2_name, ent1_type, ent2_type, ent1_id, ent2_id, distance = line
        example = MTBExamples(sent1, sent2, None, ent1_type, ent2_type, ent1_name, ent2_name, ent1_id, ent2_id)
        res.append(example)
    return res
def transform_normal_data(dataset):
    res = []
    for idx, line in enumerate(dataset[1:]):

        abstract_id, sent, ent1_name, ent2_name, ent1_type, ent2_type, ent1_id, ent2_id, distance = line
        example = InputExamples(sent, None, ent1_type, ent2_type, ent1_name, ent2_name, ent1_id, ent2_id,
                                abstract_id=abstract_id)
        res.append(example)
    return res

def process_raw_mtb_data(file_path):
    '''

    :param file_path:
    :return:
    '''
    f = open(file_path, 'r', encoding='utf-8')
    lines = f.readlines()
    f.close()
    res = []
    for idx, line in enumerate(lines[1:]):
        line = line[:-1]  # 去除换行符
        line = line.split('\t')
        abstract_id,sent1, sent2, ent1_name, ent2_name, ent1_type, ent2_type,ent1_id,ent2_id, distance = line
        example = MTBExamples(sent1, sent2, None, ent1_type, ent2_type, ent1_name, ent2_name,ent1_id,ent2_id)
        res.append(example)
    return res
def process_raw_normal_data(file_path):
    """
    这是处理predicate所需要的raw dataset
    """
    f = open(file_path, 'r', encoding='utf-8')
    lines = f.readlines()
    f.close()
    res = []
    for idx, line in enumerate(lines[1:]):
        line = line.strip()
        line = line.split('\t')
        abstract_id ,sent, ent1_name, ent2_name, ent1_type, ent2_type,ent1_id,ent2_id, distance = line
        example = InputExamples(sent, None, ent1_type, ent2_type, ent1_name, ent2_name,ent1_id,ent2_id,abstract_id=abstract_id)
        res.append(example)
    return res


def sequence_padding(inputs,length=None,value=0,seq_dims=1,mode='post'):
    '''
    这里对数据进行pad，不同的batch里面使用不同的长度
    这个方法从多个方面考虑pad，写的很高级
    这个方法一般写不出来，阿西吧


    Numpy函数，将序列padding到同一长度
    按照一个batch的最大长度进行padding
    :param inputs:(batch_size,None),每个序列的长度不一样
    :param seq_dim: 表示对哪些维度进行pad，默认为1，只有当对label进行pad的时候，seq_dim=3,因为labels.shape=(batch_size,entity_type,seq_len,seq_len)
        因为一般都是对(batch_size,seq_len)进行pad，，，
    :param length: 这个是设置补充之后的长度，一般为None，根据batch的实际长度进行pad
    :param value:
    :param mode:
    :return:
    '''
    if length is None:
        length = np.max([np.shape(x)[:seq_dims] for x in inputs],axis=0)  # length=np.array([max_batch_length])
    elif not hasattr(length,'__getitem__'): # 如果这个length的类别不是列表....,就进行转变
        length = [length]
    #logger.info('这个batch下面的最长长度为{}'.format(length[0]))

    slices = [np.s_[:length[i]] for i in range(seq_dims)]  # 获得针对针对不同维度的slice，对于seq_dims=0,slice=[None:max_len:None],max_len是seq_dims的最大值
    slices = tuple(slices) if len(slices) > 1 else slices[0]
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]  # 有多少个维数，就需要多少个(0,0),一般是一个

    outputs = []
    for x in inputs:
        # X为一个列表
        # 这里就是截取长度
        x = x[slices]
        for i in range(seq_dims):  # 对不同的维度逐步进行扩充
            if mode == 'post':
                # np.shape(x)[i]是获得当前的实际长度
                pad_width[i] = (0, length[i] - np.shape(x)[i])
            elif mode == 'pre':
                pad_width[i] = (length[i] - np.shape(x)[i], 0)
            else:
                raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, 'constant', constant_values=value)
        outputs.append(x)

    return np.array(outputs)
