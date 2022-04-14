# -*- encoding: utf-8 -*-
"""
@File    :   utils.py   
@Contact :   1329818994@qq.com
@License :   (C)Copyright 2022-2099
@Author  :   Kedaxia
@Version :   0.0.1
@Create time :   2022/4/7 18:05   
@Description :   None 

"""

import logging

from py2neo import Graph,Node,Relationship,NodeMatcher,RelationshipMatcher

class Neo4jDatabase:
    def __init__(self):
        self.graph = Graph('http://10.10.64.190:8919', auth=("neo4j", "123456"))

    def return_all_dataset(self):
        match_str = "match (n) return n"
        res = self.graph.run(match_str)
        return res.data()
    def create_node(self,ent):
        """
        构建一个生物医学实体
        label:entity_type
        id: 实体的唯一ID
        name: 这是Node属性
        """
        ent_name = ent['entity_name']
        ent_id = ent['id']
        entity_type = ent['entity_type']
        ent_norm_id = ent['norm_id']
        ent_norm_mention = ent['norm_name']
        url = ent['url']
        node = Node(ent_name,ent_id=ent_id,label=entity_type,kg_name='gastric_cancer_kg',name=ent_name,url=url,norm_id=ent_norm_id,normalize_ent_name=ent_norm_mention)
        self.graph.create(node)
    def create_relation(self,node1,node2):
        rel = Relationship(node1,'interaction',node2)
        self.graph.create(rel)
    def match_node_rels_by_name(self,name,rel_type=None,travel=None):
        pass
    def filter_node_by_entity_type(self,entity_type):
        """
        返回实体类别对应的所有节点
        """
        cypher_str = "match (n) where n.name=~'.*tum.*' return n".format(entity_type)
        res = self.graph.run(cypher_str)
        return res.data()
    def filter_node_relations_by_node(self,ent,rel=None):
        """
        ：param ent:以这个ent为起点来查找其他的关系和节点
        这个就是返回某个节点所对应的所有关系
        """
        if rel is None:
            cyoher_str = "match (n1{label:{ent_type},}) -[r]-> (n2) return n1,r,n2"
            res = self.graph.run(cyoher_str)
            return res.data()
        else:
            pass


if __name__ == '__main__':
    # 测试将已经得到的节点给
    pass