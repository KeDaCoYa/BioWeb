# -*- encoding: utf-8 -*-
"""
@File    :   multi_span.py
@Contact :   1329818994@qq.com
@License :   (C)Copyright 2022-2099
@Author  :   Kedaxia
@Version :   0.0.1
@Create time :   2022/4/4 21:11
@Description :   None

"""
# -*- encoding: utf-8 -*-

import os
import copy
import logging

import torch
import torch.nn.functional as F
from torch import nn
from ipdb import set_trace
from torch.nn.utils.rnn import pad_sequence

from src.ner.model.bert_model import BaseBert

logger = logging.getLogger("main.multiner_bertspan")

class MultiBERT_Span(BaseBert):
    def __init__(self,config):
        super(MultiBERT_Span, self).__init__(config)
        # 这个时候numtags=2，因为只有disease一种类别
        self.config = config
        # 预先设定为9
        self.num_tags = 9
        out_dims = self.bert_config.hidden_size
        mid_linear_dims = 128

        # 准备的实体类别有:DNA,RNA,Gene/Protein,Disease,Chemical/Durg,cell_type,cell_line,species
        self.chem_drug_mid_linear = nn.Sequential(
            nn.Linear(out_dims, mid_linear_dims),
            nn.Dropout(config.dropout_prob)
        )
        self.gene_protein_mid_linear = nn.Sequential(
            nn.Linear(out_dims, mid_linear_dims),
            nn.Dropout(config.dropout_prob)
        )
        self.disease_mid_linear = nn.Sequential(
            nn.Linear(out_dims, mid_linear_dims),
            nn.Dropout(config.dropout_prob)
        )
        self.cell_line_mid_linear = nn.Sequential(
            nn.Linear(out_dims, mid_linear_dims),
            nn.Dropout(config.dropout_prob)
        )
        self.cell_type_mid_linear = nn.Sequential(
            nn.Linear(out_dims, mid_linear_dims),
            nn.Dropout(config.dropout_prob)
        )
        self.dna_mid_linear = nn.Sequential(
            nn.Linear(out_dims, mid_linear_dims),
            nn.Dropout(config.dropout_prob)
        )
        self.rna_mid_linear = nn.Sequential(
            nn.Linear(out_dims, mid_linear_dims),
            nn.Dropout(config.dropout_prob)
        )
        self.species_mid_linear = nn.Sequential(
            nn.Linear(out_dims, mid_linear_dims),
            nn.Dropout(config.dropout_prob)
        )


        out_dims = 128

        #
        self.chem_drug_start_fc = nn.Linear(out_dims, self.num_tags)
        self.chem_drug_end_fc = nn.Linear(out_dims, self.num_tags)

        self.gene_protein_start_fc = nn.Linear(out_dims, self.num_tags)
        self.gene_protein_end_fc = nn.Linear(out_dims, self.num_tags)

        self.disease_start_fc = nn.Linear(out_dims, self.num_tags)
        self.disease_end_fc = nn.Linear(out_dims, self.num_tags)

        self.cell_line_start_fc = nn.Linear(out_dims, self.num_tags)
        self.cell_line_end_fc = nn.Linear(out_dims, self.num_tags)

        self.cell_type_start_fc = nn.Linear(out_dims, self.num_tags)
        self.cell_type_end_fc = nn.Linear(out_dims, self.num_tags)

        self.dna_start_fc = nn.Linear(out_dims, self.num_tags)
        self.dna_end_fc = nn.Linear(out_dims, self.num_tags)

        self.rna_start_fc = nn.Linear(out_dims, self.num_tags)
        self.rna_end_fc = nn.Linear(out_dims, self.num_tags)

        self.spec_start_fc = nn.Linear(out_dims, self.num_tags)
        self.spec_end_fc = nn.Linear(out_dims, self.num_tags)



    def forward(self, token_ids, attention_masks, token_type_ids, input_token_starts=None, start_ids=None, end_ids=None,input_true_length=None,entity_type_ids=None):
        """

        :param token_ids: 下面三个，给bert的值
        :param attention_masks:
        :param token_type_ids:
        :param input_token_starts:
        :param start_ids: 这个pad是按照batch的实际长度，并不是按照batch的subword长度，
        :param end_ids: 同上
        :param input_true_length: token_ids的真实长度
        :return:
        """

        if self.config.bert_name in ['biobert','wwm_bert','flash_quad']:
            bert_outputs = self.bert_model(input_ids=token_ids, attention_mask=attention_masks,
                                           token_type_ids=token_type_ids)
            sequence_output = bert_outputs[0]
        elif self.config.bert_name == 'kebiolm':
            bert_outputs = self.bert_model(input_ids=token_ids, attention_mask=attention_masks,
                                           token_type_ids=token_type_ids, return_dict=False)
            sequence_output = bert_outputs[2]  # shape=(batch_size,seq_len,hidden_dim)=[32, 55, 768]
        else:
            raise ValueError

        origin_sequence_output = []

        for layer, starts in zip(sequence_output, input_token_starts):
            res = layer[starts]  # shape=(seq_len,hidden_size)=(256,768)
            origin_sequence_output.append(res)

        # 这里的max_len和上面的seq_len已经不一样了，因为这里是按照token-level,而不是subword-level
        sequence_output = pad_sequence(origin_sequence_output, batch_first=True)


        if entity_type_ids[0][0].item() == 0:
            '''
            Raw text data with trained parameters
            '''
            dise_sequence_output = F.relu(self.disease_mid_linear(sequence_output)) # disease logit value
            chem_sequence_output = F.relu(self.chem_drug_mid_linear(sequence_output)) # chemical logit value
            gene_sequence_output = F.relu(self.gene_protein_mid_linear(sequence_output)) # gene/protein logit value
            spec_sequence_output = F.relu(self.species_mid_linear(sequence_output)) # species logit value
            cellline_sequence_output = F.relu(self.cell_line_mid_linear(sequence_output)) # cell line logit value
            dna_sequence_output = F.relu(self.dna_mid_linear(sequence_output)) # dna logit value
            rna_sequence_output = F.relu(self.rna_mid_linear(sequence_output)) # rna logit value
            celltype_sequence_output = F.relu(self.cell_type_mid_linear(sequence_output)) # cell type logit value

            dise_start_logits = self.disease_start_fc(dise_sequence_output) # disease logit value
            dise_end_logits = self.disease_end_fc(dise_sequence_output) # disease logit value

            chem_start_logits = self.chem_drug_start_fc(chem_sequence_output) # chemical logit value
            chem_end_logits = self.chem_drug_end_fc(chem_sequence_output) # chemical logit value

            gene_start_logits = self.gene_protein_start_fc(gene_sequence_output) # gene/protein logit value
            gene_end_logits = self.gene_protein_end_fc(gene_sequence_output) # gene/protein logit value

            spec_start_logits = self.spec_start_fc(spec_sequence_output) # species logit value
            spec_end_logits = self.spec_end_fc(spec_sequence_output) # species logit value

            cellline_start_logits = self.cell_line_start_fc(cellline_sequence_output) # cell line logit value
            cellline_end_logits = self.cell_line_end_fc(cellline_sequence_output) # cell line logit value

            dna_start_logits = self.dna_start_fc(dna_sequence_output) # dna logit value
            dna_end_logits = self.dna_end_fc(dna_sequence_output) # dna logit value

            rna_start_logits = self.rna_start_fc(rna_sequence_output) # rna logit value
            rna_end_logits = self.rna_end_fc(rna_sequence_output) # rna logit value


            celltype_start_logits = self.cell_type_start_fc(celltype_sequence_output) # cell type logit value
            celltype_end_logits = self.cell_type_end_fc(celltype_sequence_output) # cell type logit value


            # update logit and sequence_output
            sequence_output = dise_sequence_output + chem_sequence_output + gene_sequence_output + spec_sequence_output + cellline_sequence_output + dna_sequence_output + rna_sequence_output + celltype_sequence_output
            start_logits = (dise_start_logits,chem_start_logits,gene_start_logits,spec_start_logits,celltype_start_logits,cellline_start_logits,dna_start_logits,rna_start_logits)
            end_logits = (dise_end_logits,chem_end_logits,gene_end_logits,spec_end_logits,celltype_end_logits,cellline_end_logits,dna_end_logits,rna_end_logits)

        else:
            ''' 
            Train, Eval, Test with pre-defined entity type tags
            '''
            # make 1*1 conv to adopt entity type
            dise_idx = copy.deepcopy(entity_type_ids)
            chem_idx = copy.deepcopy(entity_type_ids)
            gene_idx = copy.deepcopy(entity_type_ids)
            spec_idx = copy.deepcopy(entity_type_ids)
            cellline_idx = copy.deepcopy(entity_type_ids)
            dna_idx = copy.deepcopy(entity_type_ids)
            rna_idx = copy.deepcopy(entity_type_ids)
            celltype_idx = copy.deepcopy(entity_type_ids)

            dise_idx[dise_idx != 1] = 0
            chem_idx[chem_idx != 2] = 0
            gene_idx[gene_idx != 3] = 0
            spec_idx[spec_idx != 4] = 0
            cellline_idx[cellline_idx != 5] = 0
            dna_idx[dna_idx != 6] = 0
            rna_idx[rna_idx != 7] = 0
            celltype_idx[celltype_idx != 8] = 0

            dise_sequence_output = dise_idx.unsqueeze(-1) * sequence_output
            chem_sequence_output = chem_idx.unsqueeze(-1) * sequence_output
            gene_sequence_output = gene_idx.unsqueeze(-1) * sequence_output
            spec_sequence_output = spec_idx.unsqueeze(-1) * sequence_output
            cellline_sequence_output = cellline_idx.unsqueeze(-1) * sequence_output
            dna_sequence_output = dna_idx.unsqueeze(-1) * sequence_output
            rna_sequence_output = rna_idx.unsqueeze(-1) * sequence_output
            celltype_sequence_output = celltype_idx.unsqueeze(-1) * sequence_output

            # F.tanh or F.relu
            dise_sequence_output = F.relu(self.disease_mid_linear(dise_sequence_output))  # disease logit value
            chem_sequence_output = F.relu(self.chem_drug_mid_linear(chem_sequence_output))  # chemical logit value
            gene_sequence_output = F.relu(self.gene_protein_mid_linear(gene_sequence_output))  # gene/protein logit value
            spec_sequence_output = F.relu(self.species_mid_linear(spec_sequence_output))  # species logit value
            cellline_sequence_output = F.relu(self.cell_line_mid_linear(cellline_sequence_output))  # cell line logit value
            dna_sequence_output = F.relu(self.dna_mid_linear(dna_sequence_output))  # dna logit value
            rna_sequence_output = F.relu(self.rna_mid_linear(rna_sequence_output))  # rna logit value
            celltype_sequence_output = F.relu(self.cell_type_mid_linear(celltype_sequence_output))  # cell type logit value

            dise_start_logits = self.disease_start_fc(dise_sequence_output)  # disease logit value
            dise_end_logits = self.disease_end_fc(dise_sequence_output)  # disease logit value

            chem_start_logits = self.chem_drug_start_fc(chem_sequence_output)  # chemical logit value
            chem_end_logits = self.chem_drug_end_fc(chem_sequence_output)  # chemical logit value

            gene_start_logits = self.gene_protein_start_fc(gene_sequence_output)  # gene/protein logit value
            gene_end_logits = self.gene_protein_end_fc(gene_sequence_output)  # gene/protein logit value

            spec_start_logits = self.spec_start_fc(spec_sequence_output)  # species logit value
            spec_end_logits = self.spec_end_fc(spec_sequence_output)  # species logit value

            cellline_start_logits = self.cell_line_start_fc(cellline_sequence_output)  # cell line logit value
            cellline_end_logits = self.cell_line_end_fc(cellline_sequence_output)  # cell line logit value

            dna_start_logits = self.dna_start_fc(dna_sequence_output)  # dna logit value
            dna_end_logits = self.dna_end_fc(dna_sequence_output)  # dna logit value

            rna_start_logits = self.rna_start_fc(rna_sequence_output)  # rna logit value
            rna_end_logits = self.rna_end_fc(rna_sequence_output)  # rna logit value

            celltype_start_logits = self.cell_type_start_fc(celltype_sequence_output)  # cell type logit value
            celltype_end_logits = self.cell_type_end_fc(celltype_sequence_output)  # cell type logit value

            sequence_output = dise_sequence_output + chem_sequence_output + gene_sequence_output + spec_sequence_output + cellline_sequence_output + dna_sequence_output + rna_sequence_output + celltype_sequence_output

            start_logits = dise_start_logits + chem_start_logits + gene_start_logits + spec_start_logits + celltype_start_logits + cellline_start_logits + dna_start_logits + rna_start_logits
            end_logits = dise_end_logits + chem_end_logits + gene_end_logits + spec_end_logits + celltype_end_logits + cellline_end_logits + dna_end_logits + rna_end_logits

        output = (start_logits,end_logits)



        return output

