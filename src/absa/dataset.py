"""
Dataset for Aspect-Based Sentiment Analysis (SemEval-2014 Laptop).

Prepares data for BERT-base fine-tuning.
Input format: `[CLS] sentence [SEP] aspect_term [SEP]`
"""

import logging
from typing import List, Dict, Any, Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from ..data.schema import ABSAInstance, ABSA_POLARITY_NAMES

logger = logging.getLogger(__name__)


class ABSADataset(Dataset):
    """
    PyTorch Dataset for Aspect-Based Sentiment Analysis.
    """

    def __init__(self, instances: List[ABSAInstance], tokenizer: PreTrainedTokenizer, max_length: int = 96):
        """
        Args:
            instances: List of ABSAInstance objects.
            tokenizer: HuggingFace tokenizer.
            max_length: Maximum sequence length.
        """
        self.instances = instances
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        instance = self.instances[idx]
        
        # Tokenize sentence and aspect term together
        # BERT format: [CLS] sentence [SEP] aspect_term [SEP]
        encoding = self.tokenizer(
            text=instance.sentence,
            text_pair=instance.aspect_term,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        # Remove batch dimension added by tokenizer
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
        }
        
        if 'token_type_ids' in encoding:
             item['token_type_ids'] = encoding['token_type_ids'].squeeze(0)

        item['labels'] = torch.tensor(instance.polarity_id, dtype=torch.long)
        return item
