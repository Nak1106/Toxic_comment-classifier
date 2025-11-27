from typing import Tuple

import torch
import torch.nn as nn
from transformers import AutoModel


class ContrastiveBertEncoder(nn.Module):
    """
    Siamese encoder built on top of BERT.

    It returns embeddings for each input text. The loss is defined outside.
    """

    def __init__(self, model_name: str):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size

    def encode(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state.mean(dim=1)
        return pooled

    def forward(
        self,
        input_ids_a,
        attention_mask_a,
        input_ids_b,
        attention_mask_b,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        emb_a = self.encode(input_ids_a, attention_mask_a)
        emb_b = self.encode(input_ids_b, attention_mask_b)
        return emb_a, emb_b


class ContrastiveLoss(nn.Module):
    """
    Simple margin based contrastive loss.

    label = 1 means similar (close), label = 0 means different (far).
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb_a, emb_b, labels):
        distances = torch.norm(emb_a - emb_b, p=2, dim=1)
        pos_loss = labels * distances.pow(2)
        neg_loss = (1 - labels) * torch.clamp(self.margin - distances, min=0.0).pow(2)
        loss = 0.5 * (pos_loss + neg_loss)
        return loss.mean()

