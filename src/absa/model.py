"""
Aspect-Based Sentiment Analysis Model.

Uses a pretrained BERT-base-uncased model and adds a classification head
on top of the [CLS] token for 3-class aspect polarity classification.
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

logger = logging.getLogger(__name__)


class ABSABertClassifier(nn.Module):
    """
    BERT-based aspect polarity classifier.
    """

    def __init__(self, pretrained_model_name: str = 'bert-base-uncased', num_classes: int = 3, dropout: float = 0.1):
        """
        Args:
            pretrained_model_name: Name of the pretrained BERT model.
            num_classes: Number of output classes (negative, neutral, positive).
            dropout: Dropout probability for the classification head.
        """
        super().__init__()
        self.num_classes = num_classes
        self.pretrained_model_name = pretrained_model_name
        
        logger.info(f"Initializing ABSA Model with {pretrained_model_name}")
        
        # Load pre-trained BERT
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.
            token_type_ids: Segment IDs.

        Returns:
            Logits of shape (batch_size, num_classes).
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Use the [CLS] token representation for classification
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits
