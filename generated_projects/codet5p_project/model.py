# Implement this for a project that:



import os
import glob

import torch
import torch.nn as nn

from transformers import BertModel
from transformers import BertTokenizer
from transformers import BertConfig

from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)

from transformers import (AdamW, get_linear_schedule_with_warmup)

import pandas as pd

class BertSummarizer(nn.Module):

    def __init__(self, model_type, num_layers, hidden_size, num_heads, dim_ff, dropout, vocab_size):
        super(BertSummarizer, self).__init__()

        self.bert = BertModel.from_pretrained(model_type)
        self.bert_config = BertConfig.from_pretrained(model_type)
        self.bert_config.output_hidden_states = True

        self.bert_tokenizer = BertTokenizer.from_pretrained(model_type)

        self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(hidden_size)


    def forward(self, input_ids, attention_mask):

        output_hiddens = (self.bert(input_ids, attention_mask, output_hidden_states=True))[-1]

        for i, layer in enumerate(self.layers):
            start_idx = i*self.bert_config.num_hidden_layers
            end_idx = (i+1)*self.bert_config.num_hidden_layers
            output_hiddens = layer(output_hiddens)

        output_hiddens = self.dropout(output_hiddens)

        output_hiddens = self.layer_norm(output_hiddens)

        return output_hiddens



def evaluate_model(model, dataloader, device):

    model.eval