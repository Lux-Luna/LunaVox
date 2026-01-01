"""
Abstract Frontend Base Class.

Defines the interface for all language-specific text frontends.
This enables pluggable language support without modifying core inference code.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy as np


class AbstractFrontend(ABC):
    """
    Abstract base class for language-specific text frontends.
    
    A frontend is responsible for converting text to phoneme IDs
    and extracting language-specific features (e.g., BERT embeddings for Chinese).
    """
    
    @property
    @abstractmethod
    def language(self) -> str:
        """Return the language code this frontend handles (e.g., 'en', 'zh', 'ja')."""
        pass
    
    @abstractmethod
    def tokenize(self, text: str) -> List[int]:
        """
        Convert text to a list of phoneme IDs.
        
        Args:
            text: Input text string.
            
        Returns:
            List of phoneme IDs for the text.
        """
        pass
    
    @abstractmethod
    def get_bert_features(self, text: str, phone_len: int) -> np.ndarray:
        """
        Extract BERT-style features for the given text.
        
        For languages without BERT support (English, Japanese),
        this should return a zero-filled array.
        
        Args:
            text: Input text (normalized if applicable).
            phone_len: Number of phones (to match dimensions).
            
        Returns:
            np.ndarray of shape (phone_len, BERT_FEATURE_DIM).
        """
        pass
    
    def process(self, text: str) -> Tuple[List[int], np.ndarray]:
        """
        Full text processing pipeline.
        
        Args:
            text: Input text string.
            
        Returns:
            Tuple of (phone_ids, bert_features).
        """
        ids = self.tokenize(text)
        bert = self.get_bert_features(text, len(ids))
        return ids, bert
