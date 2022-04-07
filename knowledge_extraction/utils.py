import datetime

import requests


def get_text_classification_url(text='',ip='10.10.64.190',model_name='my_tc'):
    url = 'http://{}:8950/predictions/{}/'.format(ip,model_name)
    post_data = {
        'data':text
    }

    res = requests.post(url,data=post_data)


    return eval(res.content)

def transform_entity_idx(sent_li,entities):
    """
    这个主要是将entities中的entity-level start_idx 转变为token-level
    这方便前端网页的展示
    """

    new_abstract_text = ""
    words_li = []
    for sent in sent_li:
        words_li.extend(sent)
        new_abstract_text = new_abstract_text + " ".join(sent) + " "

    new_entities = []
    for ent in entities:
        entity_level_start_idx = int(ent['start_idx'])
        entity_level_end_idx = int(ent['end_idx'])
        sent_idx = ent['pos']
        entity_name = ent['entity_name']
        pre_len = 0
        if sent_idx > 0:
            for tmp_id in range(sent_idx):
                pre_len += len(" ".join(sent_li[tmp_id])) + 1

        token_level_start_idx = len(" ".join(sent_li[sent_idx][:entity_level_start_idx]))
        token_level_end_idx = len(" ".join(sent_li[sent_idx][:entity_level_end_idx + 1]))
        if entity_level_start_idx == 0:
            begin = pre_len + token_level_start_idx
            end = pre_len + token_level_end_idx
        else:
            begin = pre_len + token_level_start_idx + 1
            end = pre_len + token_level_end_idx

        assert entity_name == new_abstract_text[begin:end], "长度不相等"

        ent['span'] = {
            "begin": begin,
            "end": end
        }
        new_entities.append(ent)
    return new_abstract_text,new_entities



def correct_datetime():
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return beijing_time
