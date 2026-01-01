"""
Abstract Frontend Base Class.

Defines the interface for all language-specific text frontends.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np


class AbstractFrontend(ABC):
    """Abstract base class for language-specific text frontends."""
    
    @property
    @abstractmethod
    def language(self) -> str:
        """Return the language code this frontend handles."""
        pass
    
    @abstractmethod
    def tokenize(self, text: str) -> List[int]:
        """Convert text to a list of phoneme IDs."""
        pass
    
    @abstractmethod
    def get_bert_features(self, text: str, phone_len: int) -> np.ndarray:
        """Extract BERT-style features for the given text."""
        pass
    
    def process(self, text: str) -> Tuple[List[int], np.ndarray]:
        """Full text processing pipeline returning (phone_ids, bert_features)."""
        ids = self.tokenize(text)
        bert = self.get_bert_features(text, len(ids))
        return ids, bert
