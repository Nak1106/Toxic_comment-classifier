from typing import Dict

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoModel


class ToxicTransformer(nn.Module):
    """
    Wrapper around a Hugging Face transformer for multi label classification.
    """

    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type="multi_label_classification",
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config,
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    @classmethod
    def from_pretrained(cls, model_name: str, num_labels: int):
        return cls(model_name, num_labels)


def create_bert_base(num_labels: int) -> ToxicTransformer:
    return ToxicTransformer("bert-base-uncased", num_labels)


def create_distilbert(num_labels: int) -> ToxicTransformer:
    return ToxicTransformer("distilbert-base-uncased", num_labels)


class LexiconHybridBert(nn.Module):
    """
    Hybrid model that combines BERT pooled output with simple lexicon features.

    The assumption is that BERT captures context and lexicon features capture
    explicit slur counts and related signals that help rare labels.
    """

    def __init__(self, model_name: str, num_labels: int, lexicon_dim: int = 3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.lexicon_dim = lexicon_dim
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + lexicon_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(self, input_ids, attention_mask, lex_feats):
        """
        input_ids: (batch, seq_len)
        attention_mask: (batch, seq_len)
        lex_feats: (batch, lexicon_dim)
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use the pooled CLS representation if available, otherwise mean pool
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            last_hidden = outputs.last_hidden_state
            pooled = last_hidden.mean(dim=1)

        x = torch.cat([pooled, lex_feats], dim=-1)
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits

    @classmethod
    def from_pretrained(cls, model_name: str, num_labels: int, lexicon_dim: int = 3):
        return cls(model_name, num_labels, lexicon_dim)

