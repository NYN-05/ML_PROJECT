"""Unified dataset loading and discovery."""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DatasetInfo:
    """Dataset metadata."""
    name: str
    path: Path
    format: str  # 'single_csv', 'fake_true_split', 'train_test'
    text_columns: List[str]
    target_column: Optional[str]
    size: int


class DatasetLoader:
    """Unified dataset loader supporting multiple formats."""
    
    LABEL_MAPPING = {
        'fake': 0, 'FAKE': 0, 'Fake': 0, '1': 0,
        'real': 1, 'REAL': 1, 'Real': 1, '0': 1,
        'true': 1, 'TRUE': 1, 'True': 1,
    }
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        
    def discover(self, data_dir: Path) -> List[DatasetInfo]:
        """Discover all datasets in directory."""
        datasets = []
        
        if not data_dir.exists():
            logger.warning(f"Data directory not found: {data_dir}")
            return datasets
            
        for item in data_dir.iterdir():
            if not item.is_dir():
                continue
                
            dataset_info = self._identify_dataset(item)
            if dataset_info:
                datasets.append(dataset_info)
                logger.info(f"Discovered: {dataset_info.name}")
                
        return datasets
    
    def _identify_dataset(self, path: Path) -> Optional[DatasetInfo]:
        """Identify dataset format and structure."""
        files = list(path.glob("*"))
        
        # Check for fake/true split
        fake_files = [f for f in files if 'fake' in f.stem.lower()]
        true_files = [f for f in files if 'true' in f.stem.lower()]
        
        if fake_files and true_files:
            return DatasetInfo(
                name=path.name,
                path=path,
                format='fake_true_split',
                text_columns=['title', 'text'],
                target_column=None,
                size=len(fake_files) + len(true_files)
            )
        
        # Check for train/test split
        train_files = [f for f in files if 'train' in f.stem.lower()]
        test_files = [f for f in files if 'test' in f.stem.lower()]
        
        if train_files and test_files:
            return DatasetInfo(
                name=path.name,
                path=path,
                format='train_test',
                text_columns=['statement', 'text'],
                target_column='label',
                size=len(train_files) + len(test_files)
            )
        
        # Check for single CSV
        csv_files = [f for f in files if f.suffix == '.csv']
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, nrows=5)
                text_cols = [c for c in df.columns if c.lower() in 
                            ['title', 'text', 'statement', 'content']]
                
                target_cols = [c for c in df.columns if c.lower() in 
                              ['label', 'target', 'class', 'label']]
                
                if text_cols:
                    return DatasetInfo(
                        name=path.name,
                        path=path,
                        format='single_csv',
                        text_columns=text_cols,
                        target_column=target_cols[0] if target_cols else None,
                        size=self._count_csv_rows(csv_file)
                    )
            except:
                continue
                
        return None
    
    def _count_csv_rows(self, path: Path) -> int:
        """Count rows in CSV efficiently."""
        try:
            return sum(1 for _ in open(path, 'r', encoding='utf-8', errors='ignore')) - 1
        except:
            return 0
    
    def load(self, dataset_info: DatasetInfo) -> pd.DataFrame:
        """Load dataset based on format."""
        
        if dataset_info.format == 'fake_true_split':
            return self._load_fake_true_split(dataset_info.path)
        elif dataset_info.format == 'train_test':
            return self._load_train_test_split(dataset_info.path)
        else:
            return self._load_single_csv(dataset_info.path)
    
    def _load_single_csv(self, path: Path) -> pd.DataFrame:
        """Load single CSV file."""
        csv_files = list(path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV found in {path}")
            
        df = pd.read_csv(csv_files[0])
        
        # Normalize labels
        if 'label' in df.columns:
            df['label'] = df['label'].map(lambda x: self.LABEL_MAPPING.get(str(x).lower(), x))
            
        return df
    
    def _load_fake_true_split(self, path: Path) -> pd.DataFrame:
        """Load fake/true split format."""
        dfs = []
        
        for f in path.glob("*.csv"):
            try:
                df = pd.read_csv(f)
                
                # Determine label from filename
                label = 0 if 'fake' in f.stem.lower() else 1
                df['label'] = label
                
                dfs.append(df)
            except:
                continue
                
        if not dfs:
            raise ValueError(f"No data loaded from {path}")
            
        return pd.concat(dfs, ignore_index=True)
    
    def _load_train_test_split(self, path: Path) -> pd.DataFrame:
        """Load train/test split format."""
        dfs = []
        
        for pattern in ["*train*", "*test*", "*valid*"]:
            for f in path.glob(pattern):
                try:
                    if f.suffix == '.tsv':
                        df = pd.read_csv(f, sep='\t')
                    else:
                        df = pd.read_csv(f)
                    dfs.append(df)
                except:
                    continue
                    
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    
    def prepare_data(self, df: pd.DataFrame, text_col: str, target_col: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training."""
        df = df.dropna(subset=[text_col, target_col])
        
        X = df[text_col].astype(str).values
        y = df[target_col].values
        
        return X, y
    
    def train_test_split(self, X: np.ndarray, y: np.ndarray, 
                        test_size: float = 0.2, random_seed: int = 42) -> Tuple:
        """Split data into train/test."""
        np.random.seed(random_seed)
        indices = np.random.permutation(len(X))
        
        split_idx = int(len(X) * (1 - test_size))
        
        train_idx, test_idx = indices[:split_idx], indices[split_idx:]
        
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


class DatasetCache:
    """Cache for processed datasets."""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get(self, key: str) -> Optional[pd.DataFrame]:
        """Get cached dataset."""
        cache_file = self.cache_dir / f"{key}.pkl"
        
        if cache_file.exists():
            import joblib
            return joblib.load(cache_file)
            
        return None
    
    def set(self, key: str, df: pd.DataFrame) -> None:
        """Cache dataset."""
        import joblib
        cache_file = self.cache_dir / f"{key}.pkl"
        joblib.dump(df, cache_file)
        
    def clear(self) -> None:
        """Clear cache."""
        for f in self.cache_dir.glob("*.pkl"):
            f.unlink()