import torch
import torch.nn as nn
from opt_einsum import contract
from long_seq import process_long_input
from losses import ATLoss
import numpy as np 
from pair_attention import Pair_attention
from transformers.models.bert.modeling_bert import  BertSelfAttention
import math
class DocREModel(nn.Module):
    def __init__(self, config, model, emb_size=768, block_size=64, num_labels=-1, num_class=97, max_mention=23, layers=1):
        super().__init__()
        self.config = config
        self.model = model
        self.hidden_size = config.hidden_size
        self.max_mention = max_mention
        self.query_size = 128
        self.pos_size = 128
        self.num_class = num_class
        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels
        self.loss_fnt = ATLoss()

        self.context_query = nn.Linear(1  * config.hidden_size,self.query_size)
        self.ent_query = nn.Linear(1  * config.hidden_size,self.query_size)
        self.head_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.tail_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.bilinear = nn.Linear(block_size * emb_size, emb_size)
        self.pair_graph = Pair_attention(config, self.emb_size, self.emb_size, layers)
        self.pos_embedding = nn.Embedding(42, self.pos_size)
        self.pair_pos = nn.Linear(emb_size + 2*self.pos_size, emb_size)

        self.dropout = nn.Dropout(0.5)
        self.predict = nn.Sequential(
            nn.Linear(self.emb_size * 3 , self.emb_size * 1),
            nn.Tanh(),
            self.dropout,
            nn.Linear(self.emb_size * 1, self.num_class)
        )

        torch.nn.init.orthogonal_(self.head_extractor.weight, gain=1)
        torch.nn.init.orthogonal_(self.tail_extractor.weight, gain=1)
        torch.nn.init.orthogonal_(self.pos_embedding.weight, gain=1)
        torch.nn.init.orthogonal_(self.bilinear.weight, gain=1)
        torch.nn.init.orthogonal_(self.pair_pos.weight, gain=1)
        torch.nn.init.orthogonal_(self.predict[0].weight, gain=1)
        torch.nn.init.orthogonal_(self.predict[-1].weight, gain=1)
    def encode(self, input_ids, attention_mask):
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
        return sequence_output, attention


    def get_hrt(self, sequence_output, attention, entity_pos, hts, ht_visible_masks, attention_mask, labels=None):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        n, h, _, c = attention.size()
        hss, tss, rss, ht_pairs, ht_evis, evis_mask, ht_context= [], [], [], [], [], [], []
        for i in range(len(entity_pos)):
            entity_embs, entity_atts, entity_masks = [], [], []
            for e in entity_pos[i]:
                context_mask = attention_mask[i] * 0
                entity_mask = torch.zeros(self.max_mention).to(sequence_output)
                entity_mask[:len(e)] = 1
                entity_masks.append(entity_mask)
                if len(e) > 1:
                    e_emb, e_att = [], []
                    for start, end in e:
                        if start + offset < c:
                            e_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])
                            context_mask[start + offset] = 1

                    if len(e_emb) > 0:
                        e_att = torch.stack(e_att, dim=0).mean(0)
                        e_emb = torch.stack(e_emb, dim=0)# m * h

                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output).unsqueeze(0)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset].unsqueeze(0)
                        e_att = attention[i, :, start + offset]
                        context_mask[start + offset] = 1
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output).unsqueeze(0)
                        e_att = torch.zeros(h, c).to(attention)
                e_emb = e_emb[:self.max_mention]
                if len(e_emb) < self.max_mention:
                    pad_mention = torch.zeros(self.config.hidden_size).to(sequence_output).unsqueeze(0).repeat(self.max_mention - len(e_emb), 1)
                    e_emb = torch.cat([e_emb, pad_mention], dim=0)
                
                entity_embs.append(e_emb)
                entity_atts.append(e_att)

            entity_masks = torch.stack(entity_masks, dim=0)
            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]

            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])
            hs_mask = torch.index_select(entity_masks, 0, ht_i[:, 0]).bool()
            ts_mask = torch.index_select(entity_masks, 0, ht_i[:, 1]).bool()

            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
            
            
            ht_att = (h_att * t_att).mean(1)
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)
            rs = contract("ld,rl->rd", sequence_output[i], ht_att)


            # P, M, H = hs.size()
           
            ht_query = self.context_query(rs).unsqueeze(1)
            hs_query = self.ent_query(hs)
            ts_query = self.ent_query(ts)
            G1 = torch.matmul(ht_query, hs_query.transpose(1,2)).squeeze(1)
            G2 = torch.matmul(ht_query, ts_query.transpose(1,2)).squeeze(1)
            G1 = G1 / math.sqrt(self.emb_size)
            G2 = G2 / math.sqrt(self.emb_size)
            G1 = G1.masked_fill_(~hs_mask, -10000.0)
            G2 = G2.masked_fill_(~ts_mask, -10000.0)
            hs_score = torch.softmax(G1, -1).unsqueeze(1)
            ts_score = torch.softmax(G2, -1).unsqueeze(1)
            hs = torch.matmul(hs_score, hs).squeeze(1)
            ts = torch.matmul(ts_score, ts).squeeze(1)
           


            hs_core = self.pos_embedding(ht_i[:, 0])
            ts_core = self.pos_embedding(ht_i[:, 1])
            

            hs = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=1)))
            ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=1)))
            
            b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
            b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)
            ht_pair = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
            ht_pair = self.bilinear(ht_pair)
            ht_pair = torch.tanh(self.pair_pos(torch.cat([hs_core, ht_pair, ts_core], dim=-1)))
            ht_visible_mask = ht_visible_masks[i].unsqueeze(0)
            ht_pair = self.pair_graph(ht_pair, ht_visible_mask)
            ht_pairs.append(ht_pair)
            hss.append(hs)
            tss.append(ts)
            rss.append(rs)
        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        rss = torch.cat(rss, dim=0)
        ht_pairs = torch.cat(ht_pairs, dim=0)
        return hss, rss, tss, ht_pairs, ht_evis, evis_mask, ht_context


    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                entity_pos=None,
                hts=None,
                ht_visible_mask=None,
                instance_mask=None,
                ):

        sequence_output, attention = self.encode(input_ids, attention_mask)

        


        hs, rs, ts, h_t, ht_evis, evis_mask, ht_context= self.get_hrt(sequence_output, attention, entity_pos, hts, ht_visible_mask, attention_mask, labels)


        logits = self.predict(torch.cat([hs, ts, h_t], dim=-1))

        output = (self.loss_fnt.get_label(logits, num_labels=self.num_labels),)
        if labels is not None:
            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(logits)
            loss = self.loss_fnt(logits.float(), labels.float())
            output = (loss.to(sequence_output),) + output
        return output
