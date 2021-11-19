import torch.nn as nn
import torch
from transformers.models.bert.modeling_bert import BertIntermediate, BertOutput, BertAttention, BertSelfAttention
import copy
from transformers.activations import ACT2FN

class Pair_attention_layer(nn.Module):
    def __init__(self, config, input_dim, output_dim, drop_p = 0.0):
        super().__init__()
        
        new_config = copy.deepcopy(config) 
        new_config.attention_probs_dropout_prob = 0
        new_config.hidden_size = input_dim
        self.attention = BertSelfAttention(new_config)
        self.LayerNorm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(drop_p)
        self.dense = nn.Linear(input_dim * 1, output_dim)
        torch.nn.init.orthogonal_(self.attention.query.weight, gain=1)
        torch.nn.init.orthogonal_(self.attention.key.weight, gain=1)
        torch.nn.init.orthogonal_(self.attention.value.weight, gain=1)
        torch.nn.init.orthogonal_(self.dense.weight, gain=1)
        self.layers = 0

    def forward(self, ht, attention_mask=None):

        sht = ht.unsqueeze(0)
        if attention_mask is None:
            extended_attention_mask=None
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        if extended_attention_mask is not None:
            extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        sht = self.attention(sht, attention_mask=extended_attention_mask)[0].squeeze()
        sht = self.dropout(torch.tanh(self.dense(sht))) + ht
        sht = self.LayerNorm(sht) 

        return sht

class Pair_attention(nn.Module):
    def __init__(self, config, input_dim, output_dim, num_layers=3):
        super().__init__()
        
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
			Pair_attention_layer(config, input_dim, output_dim)
			for _ in range(self.num_layers)
		])

    def forward(self, sht, attention_mask=None):
        
        for layer in self.layers:
            sht = layer(sht, attention_mask)
        return sht
