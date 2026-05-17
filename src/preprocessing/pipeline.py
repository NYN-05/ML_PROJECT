"""Text Preprocessing Module - Clean, tokenize, and vectorize text data."""

from __future__ import annotations

import json
import logging
import re
import string
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

logger = logging.getLogger("preprocessing")


@dataclass
class PreprocessingConfig:
    """Configuration for text preprocessing."""
    lowercase: bool = True
    remove_punctuation: bool = True
    remove_urls: bool = True
    remove_html: bool = True
    remove_numbers: bool = False
    remove_stopwords: bool = False
    stemming: bool = False
    lemmatization: bool = False
    min_word_length: int = 2
    max_features: int = 10000
    ngram_range: tuple = (1, 1)
    vectorizer_type: str = "tfidf"  # tfidf, count
    

class TextCleaner:
    """Clean and preprocess text data."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.stopwords = self._load_stopwords()
    
    def _load_stopwords(self) -> set:
        """Load English stopwords."""
        return {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
            'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her',
            'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs',
            'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
            'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
            'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
            'why', 'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
            'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's',
            't', 'can', 'will', 'just', 'don', 'should', 'now'
        }
    
    def clean_text(self, text: str) -> str:
        """Clean a single text."""
        if not isinstance(text, str) or not text:
            return ""
        
        text = text.lower() if self.config.lowercase else text
        
        if self.config.remove_html:
            text = re.sub(r'<[^>]+>', ' ', text)
        
        if self.config.remove_urls:
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
            text = re.sub(r'www\.[^\s]+', ' ', text)
        
        if self.config.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        if self.config.remove_numbers:
            text = re.sub(r'\d+', ' ', text)
        
        text = re.sub(r'\s+', ' ', text).strip()
        
        words = text.split()
        
        if self.config.remove_stopwords:
            words = [w for w in words if w not in self.stopwords]
        
        words = [w for w in words if len(w) >= self.config.min_word_length]
        
        return ' '.join(words)
    
    def clean_dataframe(self, df: pd.DataFrame, text_columns: List[str]) -> pd.DataFrame:
        """Clean text columns in a DataFrame."""
        df = df.copy()
        
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].fillna('').astype(str).apply(self.clean_text)
                logger.debug(f"Cleaned column: {col}")
        
        return df


class VectorizerWrapper:
    """TF-IDF or Count vectorizer with save/load functionality."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.vectorizer = self._create_vectorizer()
        self.is_fitted = False
    
    def _create_vectorizer(self):
        """Create the vectorizer based on config."""
        if self.config.vectorizer_type == "tfidf":
            return TfidfVectorizer(
                max_features=self.config.max_features,
                ngram_range=self.config.ngram_range,
                min_df=2,
                max_df=0.95,
                sublinear_tf=True
            )
        elif self.config.vectorizer_type == "count":
            return CountVectorizer(
                max_features=self.config.max_features,
                ngram_range=self.config.ngram_range,
                min_df=2,
                max_df=0.95
            )
        else:
            return TfidfVectorizer(
                max_features=self.config.max_features,
                ngram_range=self.config.ngram_range
            )
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit vectorizer and transform texts."""
        logger.info(f"Fitting {self.config.vectorizer_type} vectorizer on {len(texts)} texts")
        self.vectorizer.fit(texts)
        self.is_fitted = True
        result = self.vectorizer.transform(texts)
        logger.info(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        return result
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts using fitted vectorizer."""
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted yet")
        return self.vectorizer.transform(texts)
    
    def save(self, path: Path) -> None:
        """Save vectorizer to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        import joblib
        joblib.dump(self.vectorizer, path / "vectorizer.pkl")
        
        config_path = path / "vectorizer_config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
        
        logger.info(f"Saved vectorizer to: {path}")
    
    def load(self, path: Path) -> None:
        """Load vectorizer from disk."""
        import joblib
        
        self.vectorizer = joblib.load(path / "vectorizer.pkl")
        self.is_fitted = True
        
        config_path = path / "vectorizer_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = PreprocessingConfig(**json.load(f))
        
        logger.info(f"Loaded vectorizer from: {path}")


class TextPreprocessor:
    """Complete text preprocessing pipeline."""
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.cleaner = TextCleaner(self.config)
        self.vectorizer = VectorizerWrapper(self.config)
        self.text_column: Optional[str] = None
        self.target_column: Optional[str] = None
    
    def prepare_dataframe(self, df: pd.DataFrame, text_column: str, target_column: str) -> pd.DataFrame:
        """Prepare DataFrame for training."""
        self.text_column = text_column
        self.target_column = target_column
        
        df = df.copy()
        
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in DataFrame")
        
        df = df.dropna(subset=[text_column])
        
        if target_column and target_column in df.columns:
            df = df.dropna(subset=[target_column])
        
        df = self.cleaner.clean_dataframe(df, [text_column])
        
        return df
    
    def fit_vectorizer(self, texts: List[str]) -> np.ndarray:
        """Fit vectorizer on texts."""
        return self.vectorizer.fit_transform(texts)
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts using fitted vectorizer."""
        return self.vectorizer.transform(texts)
    
    def fit_transform_dataframe(self, df: pd.DataFrame) -> tuple:
        """Fit vectorizer and transform DataFrame."""
        if self.text_column is None:
            raise ValueError("Text column not set. Call prepare_dataframe first.")
        
        texts = df[self.text_column].tolist()
        X = self.fit_vectorizer(texts)
        
        if self.target_column and self.target_column in df.columns:
            y = df[self.target_column].values
            return X, y
        
        return X, None
    
    def transform_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """Transform DataFrame using fitted vectorizer."""
        if self.text_column is None:
            raise ValueError("Text column not set")
        
        texts = df[self.text_column].tolist()
        return self.transform(texts)
    
    def save(self, path: Path) -> None:
        """Save preprocessor."""
        path.mkdir(parents=True, exist_ok=True)
        
        config_path = path / "preprocessing_config.json"
        with open(config_path, 'w') as f:
            json.dump({
                "config": asdict(self.config),
                "text_column": self.text_column,
                "target_column": self.target_column
            }, f, indent=2)
        
        self.vectorizer.save(path)
        
        logger.info(f"Saved preprocessor to: {path}")
    
    def load(self, path: Path) -> None:
        """Load preprocessor."""
        config_path = path / "preprocessing_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                data = json.load(f)
                self.config = PreprocessingConfig(**data["config"])
                self.text_column = data.get("text_column")
                self.target_column = data.get("target_column")
                self.cleaner = TextCleaner(self.config)
                self.vectorizer = VectorizerWrapper(self.config)
        
        self.vectorizer.load(path)
        
        logger.info(f"Loaded preprocessor from: {path}")


def create_preprocessor(
    lowercase: bool = True,
    remove_punctuation: bool = True,
    remove_urls: bool = True,
    remove_stopwords: bool = False,
    max_features: int = 10000,
    ngram_range: tuple = (1, 2),
    vectorizer_type: str = "tfidf"
) -> TextPreprocessor:
    """Create a preprocessor with custom configuration."""
    config = PreprocessingConfig(
        lowercase=lowercase,
        remove_punctuation=remove_punctuation,
        remove_urls=remove_urls,
        remove_stopwords=remove_stopwords,
        max_features=max_features,
        ngram_range=ngram_range,
        vectorizer_type=vectorizer_type
    )
    return TextPreprocessor(config)