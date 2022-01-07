import torch
import torch.nn as nn


class BertPrompt(nn.Module):
    def __init__(self, model, p_neg, p_pos, mask_id):
        super().__init__()
        self.model = model
        self.p_neg = p_neg
        self.p_pos = p_pos
        self.mask_id = mask_id

    def forward(self, input_ids, attention_mask):
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        pos = (input_ids == self.mask_id)
        logits = logits[pos]
        return torch.cat([logits[:, self.p_neg].unsqueeze(-1), logits[:, self.p_pos].unsqueeze(-1)], dim=-1)


class Bert(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        return logits
