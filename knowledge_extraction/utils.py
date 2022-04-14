import datetime

import requests


def get_text_classification_url(text='', ip='10.10.64.190', model_name='my_tc'):
    url = 'http://{}:8950/predictions/{}/'.format(ip, model_name)
    post_data = {
        'data': text
    }

    res = requests.post(url, data=post_data)

    return eval(res.content)


def transform_triplets_info(entities, triplets):
    """
    将关系抽取模型得到的结果给丰富一下，用于之后的可视化
    同时将triplets的关系类别丰富一下，由1变成PPI,DDI,CPI,GDI,....
    """
    new_triplets = []
    for triple in triplets:
        e1_id = triple['e1_id']
        e2_id = triple['e2_id']
        ent1, ent2 = None,None
        for ent in entities:
            if ent1 and ent2:
                break
            if ent['id'] == e1_id:
                ent1 = ent
            if ent['id'] == e2_id:
                ent2 = ent
        ent1_name = ent1['entity_name']
        ent1_type = ent1['entity_type']
        ent1_pos = ent1['pos']
        ent1_norm_name = ent1['norm_name']
        ent1_norm_id = ent1['norm_id']

        ent2_name = ent2['entity_name']
        ent2_type = ent2['entity_type']
        ent2_pos = ent2['pos']
        ent2_norm_name = ent2['norm_name']
        ent2_norm_id = ent2['norm_id']

        rel_id = triple['id']
        relation_type = triple['relation_type']
        if (ent1_type, ent2_type) in [('Gene/Protein', 'Gene/Protein'), ('DNA', 'Gene/Protein'),
                                      ('Gene/Protein', 'DNA')]:
            relation_type = 'Gene/Protein-Gene/Protein Interaction'
        elif (ent1_type, ent2_type) in [('Chemical/Drug', 'Gene/Protein'), ('Gene/Protein', 'Chemical/Drug')]:
            relation_type = 'Chemical/Drug-Gene/Protein Interaction'
        elif (ent1_type, ent2_type) in [("Gene/Protein", "Disease"), ("Disease", "Gene/Protein")]:
            relation_type = 'Gene/Protein-Disease Interaction'
        elif (ent1_type, ent2_type) in [("Chemical/Drug", "Disease"), ("Disease", "Chemical/Drug")]:
            relation_type = 'Drug-Disease Interaction'
        elif (ent1_type, ent2_type) in [("Chemical/Drug", "Chemical/Drug")]:
            relation_type = 'Drug-Drug Interaction'
        else:
            pass
        new_triplets.append({
            'rel_id': rel_id,
            'rel_type': relation_type,
            'ent1_name': ent1_name,
            'ent1_type': ent1_type,
            'ent1_pos': ent1_pos,
            'ent1_norm_name': ent1_norm_name,
            'ent1_norm_id': ent1_norm_id,
            'ent2_name': ent2_name,
            'ent2_type': ent2_type,
            'ent2_pos': ent2_pos,
            'ent2_norm_name': ent2_norm_name,
            'ent2_norm_id': ent2_norm_id
        })
    return new_triplets
def transform_entity_idx(sent_li, entities):
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
    return new_abstract_text, new_entities

def correct_datetime():
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return beijing_time
