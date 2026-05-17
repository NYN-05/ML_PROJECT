"""Unified text preprocessing pipeline."""

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Preprocessing configuration."""
    lowercase: bool = True
    remove_urls: bool = True
    remove_html: bool = True
    remove_punctuation: bool = False
    remove_numbers: bool = False
    remove_extra_whitespace: bool = True
    min_word_length: int = 2
    max_features: int = 10000
    ngram_range: Tuple[int, int] = (1, 2)
    use_tfidf: bool = True
    sublinear_tf: bool = True


class TextCleaner:
    """Text cleaning utilities."""
    
    URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
    HTML_PATTERN = re.compile(r'<.*?>')
    EMAIL_PATTERN = re.compile(r'\S+@\S+\.\S+')
    MULTI_SPACE = re.compile(r'\s+')
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        
    def clean(self, text: str) -> str:
        """Clean single text."""
        if not isinstance(text, str):
            return ""
            
        # Lowercase
        if self.config.lowercase:
            text = text.lower()
            
        # Remove URLs
        if self.config.remove_urls:
            text = self.URL_PATTERN.sub(' ', text)
            
        # Remove HTML
        if self.config.remove_html:
            text = self.HTML_PATTERN.sub(' ', text)
            
        # Remove emails
        text = self.EMAIL_PATTERN.sub(' ', text)
        
        # Remove punctuation
        if self.config.remove_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)
            
        # Remove numbers
        if self.config.remove_numbers:
            text = re.sub(r'\d+', ' ', text)
            
        # Remove extra whitespace
        if self.config.remove_extra_whitespace:
            text = self.MULTI_SPACE.sub(' ', text).strip()
            
        # Filter short words
        if self.config.min_word_length > 2:
            words = text.split()
            words = [w for w in words if len(w) >= self.config.min_word_length]
            text = ' '.join(words)
            
        return text
    
    def clean_batch(self, texts: List[str]) -> List[str]:
        """Clean batch of texts."""
        return [self.clean(t) for t in texts]


class TextVectorizer:
    """Text vectorization using TF-IDF."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.vectorizer = None
        
    def fit(self, texts: List[str]) -> 'TextVectorizer':
        """Fit vectorizer on texts."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        self.vectorizer = TfidfVectorizer(
            max_features=self.config.max_features,
            ngram_range=self.config.ngram_range,
            sublinear_tf=self.config.sublinear_tf,
            min_df=2,
            max_df=0.95
        )
        
        self.vectorizer.fit(texts)
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to vectors."""
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted")
            
        return self.vectorizer.transform(texts).toarray()
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(texts)
        return self.transform(texts)
    
    def save(self, path: str) -> None:
        """Save vectorizer."""
        import joblib
        joblib.dump(self.vectorizer, path)
        
    def load(self, path: str) -> None:
        """Load vectorizer."""
        import joblib
        self.vectorizer = joblib.load(path)


class TextPreprocessor:
    """Complete text preprocessing pipeline."""
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.cleaner = TextCleaner(self.config)
        self.vectorizer = TextVectorizer(self.config)
        
    def preprocess(self, texts: List[str]) -> np.ndarray:
        """Complete preprocessing pipeline."""
        # Clean texts
        cleaned = self.cleaner.clean_batch(texts)
        
        # Vectorize
        vectors = self.vectorizer.fit_transform(cleaned)
        
        logger.info(f"Preprocessed {len(texts)} texts, {vectors.shape[1]} features")
        
        return vectors
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform new texts using fitted vectorizer."""
        cleaned = self.cleaner.clean_batch(texts)
        return self.vectorizer.transform(cleaned)
    
    def save_vectorizer(self, path: str) -> None:
        """Save fitted vectorizer."""
        self.vectorizer.save(path)
        
    def load_vectorizer(self, path: str) -> None:
        """Load fitted vectorizer."""
        self.vectorizer.load(path)


class Tokenizer:
    """Tokenizer for LSTM models."""
    
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}
        self.word_counts = {}
        
    def fit(self, texts: List[str]) -> 'Tokenizer':
        """Build vocabulary from texts."""
        for text in texts:
            words = text.lower().split()
            for word in words:
                self.word_counts[word] = self.word_counts.get(word, 0) + 1
        
        # Keep top vocab_size words
        sorted_words = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)
        
        idx = 2
        for word, _ in sorted_words[:self.vocab_size - 2]:
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
            idx += 1
            
        return self
    
    def encode(self, text: str, max_length: int = 256) -> List[int]:
        """Encode text to indices."""
        words = text.lower().split()
        indices = [self.word_to_idx.get(w, self.word_to_idx['<UNK>']) for w in words]
        
        # Pad or truncate
        if len(indices) < max_length:
            indices += [0] * (max_length - len(indices))
        else:
            indices = indices[:max_length]
            
        return indices
    
    def decode(self, indices: List[int]) -> str:
        """Decode indices to text."""
        words = [self.idx_to_word.get(i, '<UNK>') for i in indices]
        return ' '.join([w for w in words if w not in ['<PAD>', '<UNK>']])
    
    def save(self, path: str) -> None:
        """Save tokenizer."""
        import joblib
        joblib.dump({
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word
        }, path)
        
    def load(self, path: str) -> None:
        """Load tokenizer."""
        import joblib
        data = joblib.load(path)
        self.word_to_idx = data['word_to_idx']
        self.idx_to_word = data['idx_to_word']
        self.vocab_size = len(self.word_to_idx)