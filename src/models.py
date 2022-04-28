#!/usr/bin/env python
import torch
from torch import nn
from transformers import AutoModel


class RumourCLS(nn.Module):
    def __init__(self, pre_encoder):

        super(RumourCLS, self).__init__()
        self.encoder = AutoModel.from_pretrained(pre_encoder)
        hidden_size = self.encoder.config.hidden_size
        self.cls = nn.Linear(hidden_size, 2)
        # self.gru = nn.GRUCell(hidden_size, hidden_size)
        # self.evidence_cls = nn.Linear(hidden_size * 2, 2)

    def forward(self, reps, masks):
        texts_emb = self.encoder(input_ids=reps, attention_mask=masks).last_hidden_state
        # first token
        texts_emb = texts_emb[:, 0, :]
        logits = self.cls(texts_emb)

        # gru
        # hiddens = []
        # input_ids = reps.tolist()
        # for i in range(reps.size(0)):
        #     hidden = texts_emb[i, 0, :].view(1, -1)
        #     for jidx, j in enumerate(input_ids[0]):
        #         if j == 2:
        #             hidden = self.gru(texts_emb[i, jidx, :].view(1, -1), hidden)
        #     hiddens.append(hidden)
        # logits = self.cls(torch.cat(hiddens, 0))

        # evidence
        # logits = []
        # input_ids = reps.tolist()
        # for i in range(reps.size(0)):
        #     sent_sep_index = []
        #     for jidx, j in enumerate(input_ids[0]):
        #         if j == 2:
        #             sent_sep_index.append(jidx)
        #     sent_rep = torch.index_select(texts_emb[i, :, :], 0, torch.Tensor(sent_sep_index).type_as(reps).long())
        #     first_sent = sent_rep[0].view(1, -1).repeat(sent_rep.size(0), 1)
        #     logit = self.evidence_cls(torch.cat([first_sent, sent_rep], 1))
        #     logits.append(torch.mean(logit, 0))
        # logits = torch.stack(logits, 0)

        return logits

