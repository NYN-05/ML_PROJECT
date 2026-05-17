"""Dataset Loader - Load individual datasets with proper splitting."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from .dataset_discovery import DatasetInfo

logger = logging.getLogger("data.dataset_loader")


@dataclass
class LoadedDataset:
    """Loaded dataset with splits."""
    train: Optional[pd.DataFrame]
    test: Optional[pd.DataFrame]
    validation: Optional[pd.DataFrame]
    metadata: DatasetInfo


class DatasetLoader:
    """Load and split datasets for training."""
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
    
    def load(self, dataset_info: DatasetInfo, test_size: float = 0.2, validation_size: float = 0.1) -> LoadedDataset:
        """Load a dataset and split into train/test/validation."""
        logger.info(f"Loading dataset: {dataset_info.name}")
        
        if dataset_info.dataset_type == "fake_true_split":
            return self._load_fake_true_split(dataset_info)
        elif dataset_info.dataset_type == "train_test_split":
            return self._load_train_test_split(dataset_info)
        elif dataset_info.single_file and dataset_info.split_column:
            return self._load_with_split_column(dataset_info)
        elif dataset_info.single_file:
            return self._load_single_file(dataset_info, test_size, validation_size)
        else:
            logger.warning(f"Unknown dataset type for: {dataset_info.name}")
            return LoadedDataset(None, None, None, dataset_info)
    
    def _load_fake_true_split(self, dataset_info: DatasetInfo) -> LoadedDataset:
        """Load Fake.csv/True.csv format."""
        fake_dfs = []
        real_dfs = []
        
        if dataset_info.train_file:
            df = self._safe_read_csv(dataset_info.train_file)
            if df is not None:
                filename = dataset_info.train_file.name.lower()
                if 'fake' in filename:
                    df = self._add_label(df, is_fake=True)
                    fake_dfs.append(df)
                else:
                    df = self._add_label(df, is_fake=False)
                    real_dfs.append(df)
        
        if dataset_info.test_file:
            df = self._safe_read_csv(dataset_info.test_file)
            if df is not None:
                filename = dataset_info.test_file.name.lower()
                if 'fake' in filename:
                    df = self._add_label(df, is_fake=True)
                    fake_dfs.append(df)
                else:
                    df = self._add_label(df, is_fake=False)
                    real_dfs.append(df)
        
        all_dfs = fake_dfs + real_dfs
        if not all_dfs:
            return LoadedDataset(None, None, None, dataset_info)
        
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        logger.info(f"  Loaded fake_true_split: total={len(combined_df)}")
        
        return LoadedDataset(combined_df, None, None, dataset_info)
    
    def _load_train_test_split(self, dataset_info: DatasetInfo) -> LoadedDataset:
        """Load train/test CSV files."""
        train_df = self._safe_read_csv(dataset_info.train_file) if dataset_info.train_file else None
        test_df = self._safe_read_csv(dataset_info.test_file) if dataset_info.test_file else None
        validation_df = self._safe_read_csv(dataset_info.validation_file) if dataset_info.validation_file else None
        
        logger.info(f"  Loaded train_test_split: train={len(train_df) if train_df is not None else 0}, test={len(test_df) if test_df is not None else 0}")
        
        return LoadedDataset(train_df, test_df, validation_df, dataset_info)
    
    def _load_with_split_column(self, dataset_info: DatasetInfo) -> LoadedDataset:
        """Load single CSV with split column."""
        df = self._safe_read_csv(dataset_info.single_file)
        
        if df is None or dataset_info.split_column not in df.columns:
            return self._load_single_file(dataset_info, 0.2, 0.1)
        
        split_col = dataset_info.split_column
        
        train_df = df[df[split_col].astype(str).str.lower().isin(['train', 'training', '1', 'true'])].drop(columns=[split_col])
        test_df = df[df[split_col].astype(str).str.lower().isin(['test', 'testing', '0', 'false'])].drop(columns=[split_col])
        
        val_df = None
        val_mask = df[split_col].astype(str).str.lower().isin(['validation', 'valid', 'dev'])
        if val_mask.any():
            val_df = df[val_mask].drop(columns=[split_col])
        
        logger.info(f"  Loaded with split column: train={len(train_df)}, test={len(test_df)}")
        
        return LoadedDataset(train_df, test_df, val_df, dataset_info)
    
    def _load_single_file(self, dataset_info: DatasetInfo, test_size: float, validation_size: float) -> LoadedDataset:
        """Load single CSV and split into train/test/validation."""
        df = self._safe_read_csv(dataset_info.single_file)
        
        if df is None:
            return LoadedDataset(None, None, None, dataset_info)
        
        target_col = dataset_info.target_column
        
        if target_col and target_col in df.columns:
            stratify = df[target_col]
        else:
            stratify = None
        
        test_val_size = test_size + validation_size
        train_df, temp_df = train_test_split(
            df, 
            test_size=test_val_size,
            random_state=self.random_seed,
            stratify=stratify
        )
        
        if validation_size > 0:
            val_size = validation_size / test_val_size
            val_df, test_df = train_test_split(
                temp_df,
                test_size=1 - val_size,
                random_state=self.random_seed
            )
        else:
            val_df = None
            test_df = temp_df
        
        logger.info(f"  Loaded single file: train={len(train_df)}, test={len(test_df)}")
        
        return LoadedDataset(train_df, test_df, val_df, dataset_info)
    
    def _add_label(self, df: pd.DataFrame, is_fake: bool) -> pd.DataFrame:
        """Add label column to Fake/True dataset."""
        df = df.copy()
        df['label'] = 'FAKE' if is_fake else 'REAL'
        return df
    
    def _safe_read_csv(self, path: Path) -> Optional[pd.DataFrame]:
        """Safely read CSV file."""
        for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
            try:
                return pd.read_csv(path, encoding=encoding, on_bad_lines='skip')
            except Exception:
                continue
        return None


def load_dataset(dataset_info: DatasetInfo, test_size: float = 0.2, validation_size: float = 0.1) -> LoadedDataset:
    """Convenience function to load a dataset."""
    loader = DatasetLoader()
    return loader.load(dataset_info, test_size, validation_size)