import logging
import json
import time
from collections import defaultdict

from ipdb import set_trace
from nltk import sent_tokenize, wordpunct_tokenize


from django.shortcuts import render
from django.http import HttpResponse
from django.views import View


from src.function_utils import get_logger
from src.ner.model.ner_model import MyNERModel
from src.ner.ner_predicate import get_ner_bert_config
from src.ner.utils.preprocess_utils import get_text_data_from_pubmed, generate_re_normal
from knowledge_extraction.models import PmidAnnotate
from knowledge_extraction.utils import get_text_classification_url, transform_entity_idx, correct_datetime
from src.norm.models.normalize_model import MyNormModel
from src.norm.norm_predicate import get_norm_bert_config, init_norm_model
from src.re.dataset_utils.preprocess_utils import process_raw_normal_data
from src.re.model.re_model import MyReModel
from src.re.re_predicate import get_re_bert_config, init_re_model

logger = get_logger()

class Test(View):
    logger.info("这是其他的view初始化")
    def get(self,request,*args,**kwargs):
        self.hello_()
        print(request.GET)
        return HttpResponse("get_Test")


    def hello_(self):
        print("Hello_world")


class KnowledgeExtraction(View):

    ner_ckpt_path = "./src/trained_models/ner/bert_span.pt"
    logger.info("加载NER模型....")
    config = get_ner_bert_config()
    model = MyNERModel(ner_ckpt_path,config)
    logger.info("!!加载NER模型完成!!")

    logger.info("开始加载RE模型....")
    config = get_re_bert_config()
    re_ckpt_path = "./src/trained_models/re/single/rbert/model.pt"
    re_model = MyReModel(re_ckpt_path, config)
    logger.info("!!加载RE模型完成!!")

    logger.info("开始加载Normalize模型....")
    config = get_norm_bert_config()
    nrom_ckpt_path = '/root/code/bioner/embedding/SapBERT-from-PubMedBERT-fulltext'
    norm_model = MyNormModel(nrom_ckpt_path, config)
    logger.info("!!加载Norm模型完成!!")

    def get(self,request,*args,**kwargs):
        print(request.GET)
        outputs = {}
        all_triplets = None
        if request.GET['req_type'] == 'pmid':
            pmid = request.GET['pmid']
            start = time.time()
            query = PmidAnnotate.objects.filter(pmid=pmid)
            logger.info("数据库查询花费时间:{:.3f}".format(time.time()-start))
            if len(query) == 1:
                obj = query[0]
                outputs['entities'] = eval(obj.entities)
                outputs['triplets'] = eval(obj.relations)
                logger.info("数据库中有此文章，直接查询返回")
                return HttpResponse(json.dumps(outputs, ensure_ascii=False),
                                    content_type="application/json;charset=utf-8")

            # 首先到数据库进行查询

            abstract_text = get_text_data_from_pubmed(pmid)
        else:
            pmid = ""
            abstract_text = request.GET['sample_text']

        sents = sent_tokenize(abstract_text)
        sent_li = []
        for sent in sents:
            sent_li.append(wordpunct_tokenize(sent))

        if request.GET['task_type'] == 'entity_extraction':

            start = time.time()
            entities = self.entity_extraction(abstract_text)
            print("实体抽取花费时间:{}s".format(time.time() - start))


        elif request.GET['task_type'] == 'entity_extraction,entity_normalization':

            start1 = time.time()
            entities = self.entity_extraction(abstract_text)
            print("实体抽取花费时间:{}s".format(time.time() - start1))
            start2 = time.time()
            entities = self.entity_normalization(entities)
            print("实体标准化花费时间:{}s".format(time.time() - start2))
            print("最终花费时间:{}s".format(time.time() - start1))



        elif request.GET['task_type'] == 'entity_extraction,entity_normalization,relation_extraction':

            start1 = time.time()
            entities = self.entity_extraction(abstract_text)
            self.reprots(entities)
            print("实体抽取花费时间:{}s".format(time.time() - start1))

            start2 = time.time()
            entities = self.entity_normalization(entities)
            print("实体标准化花费时间:{}s".format(time.time() - start2))

            start3 = time.time()
            all_dataset = generate_re_normal(pmid, sent_li, entities)
            all_triplets = self.relation_extraction(all_dataset)
            logger.info("生成数据个数:{}的最终三元组个数:{}".format(len(all_dataset),len(all_triplets)))
            print("关系抽取花费时间:{}s".format(time.time() - start3))
            print("全部花费时间:{}s".format(time.time() - start1))
            outputs['triplets'] = all_triplets

        else:
            raise ValueError("没有此选项")

        abstract_text, entities_li = transform_entity_idx(sent_li, entities)
        if request.GET['req_type'] == 'pmid':
            now_ = correct_datetime()
            new_obj = PmidAnnotate(pmid=pmid,entities=str(entities),relations=str(all_triplets),update_time=now_,abstract_text=abstract_text)
            new_obj.save()
            logger.info("保存PMID:{}到MySQL成功".format(pmid))
        # 转变entites为字典形式存贮，entity id为key
        entities_dict = {}
        for ent in entities_li:
            entities_dict[ent['id']] = ent
        if "triplets" in outputs:
            for triple in all_triplets:
                e1_id = triple['e1_id']
                e2_id = triple['e2_id']
                print("三员组:{}:{}-{}:{}".format(entities_dict[e1_id]['entity_name'],entities_dict[e1_id]['entity_type'],entities_dict[e2_id]['entity_name'],entities_dict[e2_id]['entity_type']))

        outputs['entities'] = entities
        outputs['abstract_text'] = abstract_text
        # for ent in entities:
        #     print(ent)
        return HttpResponse(json.dumps(outputs, ensure_ascii=False), content_type="application/json;charset=utf-8")
    def post(self,request,*args,**kwargs):
        pass
    def reprots(self,entities):
        """
        对这次抽取的entities进行统计
        """
        logger.info("实体抽取个数为:{}".format(len(entities)))

        counter = defaultdict(int)
        for ent in entities:
            counter[ent['entity_type']] += 1
        logger.info(counter)
    def entity_extraction(self,abstract_text):

        entities = self.model.entity_extraction(abstract_text)

        return entities

    def entity_normalization(self,entities):

        new_entities = self.norm_model.entity_normalization(entities)
        return new_entities
    def relation_extraction(self,all_dataset):
        """
        :param all_dataset:这个就是关系分类的数据格式类被
        """
        #all_dataset = process_raw_normal_data("./src/re/normal.txt")
        all_triplets = self.re_model.relation_extraction(all_dataset)
        for triple in all_triplets:
            print(triple)
        return all_triplets

#
# class EntityNormalization(View):
#     logger.info("加载Normalize模型....")
#     config = get_norm_bert_config()
#     ckpt_path = '/root/code/bioner/embedding/SapBERT-from-PubMedBERT-fulltext'
#
#     norm_model = MyNormModel(ckpt_path,config)
#
#     logger.info("!!加载Norm模型完成!!")
#     def post(self,request,*args,**kwargs):
#         return HttpResponse("Json")
#     def get(self,request,*args,**kwargs):
#         entities = [
#             {"entity_type": "Gene/Protein", "start_idx": "4", "end_idx": "4", "entity_name": "TWIST", "pos": 9, "id": "e31"},
#
#             {"entity_type": "cell_type", "start_idx": "10", "end_idx": "12", "entity_name": "gastric cancer cell", "pos": 9, "id": "e33"},
#
#         ]
#         start = time.time()
#         new_entities = self.norm_model.entity_normalization(entities)
#         for ent in new_entities:
#             print(ent['entity_name'],ent['norm_id'],ent['norm_name'])
#         logger.info("实体抽取花费时间:{:.3f}".format(time.time()-start))
#         return HttpResponse("这是Norm")
#
#
# class RelationExtraction(View):
#
#     logger.info("加载RE模型....")
#     config = get_re_bert_config()
#     ckpt_path = "./src/trained_models/re/single/rbert/model.pt"
#     re_model = MyReModel(ckpt_path,config)
#
#     logger.info("!!加载RE模型完成!!")
#
#     def get(self,request,*args,**kwargs):
#         all_dataset = process_raw_normal_data("./src/re/normal.txt")
#         all_triplets = self.re_model.relation_extraction(all_dataset)
#         for triple in all_triplets:
#             print(triple)
#
#         return HttpResponse("这是关系抽取模型的get")
#
#
# class EntityExtraction(View):
#     ckpt_path = "./src/trained_models/ner/bert_span.pt"
#     logger.info("加载NER模型....")
#     config = get_ner_bert_config()
#     # model = MyNERModel(ckpt_path,config)
#     logger.info("!!加载NER模型完成!!")
#
#     def get(self,request,pmid,*args,**kwargs):
#         abstract_text = get_text_data_from_pubmed(pmid)
#         start = time.time()
#
#         entities = self.model.entity_extraction(abstract_text)
#         for ent in entities:
#             print(ent)
#
#         print("实体抽取花费时间:{}s".format(time.time()-start))
#
#         return HttpResponse(abstract_text)
#


def index(request):
    return render(request,"index.html")


def entity_extraction(request):
    return HttpResponse("good")
def relation_extraction(request):
    return HttpResponse("这是get的relation_extraction")
def entity_normalization(request):
    return HttpResponse("这是get的entity_normalization")

def get_text_classification(request):
    '''
    这是demo展示，文本分类
    '''
    print(request.POST)
    print('----------------------')
    raw_text =  request.POST['raw_text'][0]

    res = get_text_classification_url(text=raw_text)

    # 这是返回json格式(ajax)
    data = {}
    data['success'] = True
    data['res'] = res
    # 以Ajax的形式返回
    return HttpResponse(json.dumps(data, ensure_ascii=False), content_type="application/json;charset=utf-8")



