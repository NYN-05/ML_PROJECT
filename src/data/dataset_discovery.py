"""Dataset Discovery System - Automatic recursive dataset scanning and detection."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Literal

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger("data.dataset_discovery")


DatasetType = Literal["fake_true_split", "train_test_split", "single_csv", "tsv_split", "unknown"]


@dataclass
class DatasetInfo:
    """Information about a discovered dataset."""
    name: str
    path: Path
    dataset_type: DatasetType
    train_file: Optional[Path] = None
    test_file: Optional[Path] = None
    validation_file: Optional[Path] = None
    single_file: Optional[Path] = None
    target_column: Optional[str] = None
    text_columns: List[str] = field(default_factory=list)
    n_rows: int = 0
    n_cols: int = 0
    detected_encoding: str = "utf-8"
    split_column: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d["path"] = str(self.path)
        d["train_file"] = str(self.train_file) if self.train_file else None
        d["test_file"] = str(self.test_file) if self.test_file else None
        d["validation_file"] = str(self.validation_file) if self.validation_file else None
        d["single_file"] = str(self.single_file) if self.single_file else None
        return d


class DatasetDiscovery:
    """Automatic dataset discovery system that recursively scans folders."""
    
    TRAIN_NAMES = {
        'train.csv', 'training.csv', 'train', 'traindata.csv', 'trainingdata.csv',
        'train.tsv', 'training.tsv'
    }
    TEST_NAMES = {
        'test.csv', 'testing.csv', 'test', 'testdata.csv', 'testingdata.csv',
        'test.tsv', 'testing.tsv'
    }
    VALIDATION_NAMES = {
        'validation.csv', 'valid.csv', 'validation.tsv', 'valid.tsv',
        'dev.csv', 'dev.tsv'
    }
    FAKE_TRUE_NAMES = {
        'fake.csv', 'true.csv', 'fake.tsv', 'true.tsv',
        'fake', 'true', 'Fake.csv', 'True.csv'
    }
    DATA_NAMES = {
        'data.csv', 'dataset.csv', 'data.tsv', 'dataset.tsv',
        'data', 'dataset', 'all.csv', 'combined.csv', 'news.csv'
    }
    
    TARGET_NAMES = {
        'target', 'label', 'class', 'y', 'output', 'prediction',
        'target_label', 'class_label', 'label_raw', 'Label'
    }
    
    TEXT_COLUMNS = {
        'text', 'article', 'content', 'body', 'statement', 'claim',
        'title', 'headline', 'news', 'statement_text'
    }
    
    def __init__(self, root_path: str | Path):
        self.root_path = Path(root_path)
        self.datasets: List[DatasetInfo] = []
        self.exclude_dirs = {'output', 'outputs', 'processed', 'logs', 'temp', 'tmp', '__pycache__', '.git', 'node_modules'}
    
    def discover(self) -> List[DatasetInfo]:
        """Discover all datasets in the root path."""
        logger.info(f"Discovering datasets in: {self.root_path}")
        
        if not self.root_path.exists():
            logger.error(f"Root path does not exist: {self.root_path}")
            return []
        
        subdirs = [d for d in self.root_path.iterdir() 
                   if d.is_dir() and d.name.lower() not in self.exclude_dirs]
        
        logger.info(f"Found {len(subdirs)} potential dataset directories")
        
        for subdir in tqdm(subdirs, desc="Scanning datasets"):
            dataset_info = self._analyze_dataset(subdir)
            if dataset_info:
                self.datasets.append(dataset_info)
                logger.info(f"  Found: {dataset_info.name} ({dataset_info.dataset_type})")
        
        logger.info(f"Discovered {len(self.datasets)} valid datasets")
        return self.datasets
    
    def _analyze_dataset(self, dataset_path: Path) -> Optional[DatasetInfo]:
        """Analyze a single dataset directory."""
        csv_files = list(dataset_path.glob("*.csv"))
        tsv_files = list(dataset_path.glob("*.tsv"))
        all_files = csv_files + tsv_files
        
        if not all_files:
            return None
        
        file_map = {f.name.lower(): f for f in all_files}
        
        train_file, test_file, validation_file, single_file, split_column = self._detect_format(file_map, all_files)
        
        if single_file is None and train_file is None and test_file is None:
            fake_file, true_file = self._detect_fake_true_split(file_map, all_files)
            if fake_file and true_file:
                train_file = fake_file
                test_file = true_file
        
        if single_file is None and train_file is None and test_file is None:
            single_file = self._detect_any_csv_with_label(all_files)
        
        if single_file is None and train_file is None and test_file is None:
            return None
        
        sample_df = self._load_sample(train_file, test_file, single_file)
        if sample_df is None or sample_df.empty:
            return None
        
        target_col = self._detect_target_column(sample_df)
        text_cols = self._detect_text_columns(sample_df)
        
        dataset_type = self._get_dataset_type(train_file, test_file, single_file, split_column)
        
        n_rows = self._count_total_rows(train_file, test_file, validation_file, single_file)
        
        return DatasetInfo(
            name=dataset_path.name,
            path=dataset_path,
            dataset_type=dataset_type,
            train_file=train_file,
            test_file=test_file,
            validation_file=validation_file,
            single_file=single_file,
            target_column=target_col,
            text_columns=text_cols,
            n_rows=n_rows,
            n_cols=len(sample_df.columns)
        )
    
    def _detect_format(self, file_map: Dict[str, Path], all_files: List[Path]) -> tuple:
        """Detect train/test/validation split format."""
        train_file = None
        test_file = None
        validation_file = None
        single_file = None
        split_column = None
        
        for name in self.TRAIN_NAMES:
            if name in file_map:
                train_file = file_map[name]
                break
        
        for name in self.TEST_NAMES:
            if name in file_map:
                test_file = file_map[name]
                break
        
        for name in self.VALIDATION_NAMES:
            if name in file_map:
                validation_file = file_map[name]
                break
        
        for name in self.DATA_NAMES:
            if name in file_map:
                single_file = file_map[name]
                break
        
        if single_file is not None:
            df = self._safe_read(single_file)
            if df is not None:
                split_column = self._detect_split_column(df)
        
        return train_file, test_file, validation_file, single_file, split_column
    
    def _detect_fake_true_split(self, file_map: Dict[str, Path], all_files: List[Path]) -> tuple:
        """Detect Fake.csv/True.csv split format."""
        fake_file = None
        true_file = None
        
        for fname, fpath in file_map.items():
            if 'fake' in fname and 'true' not in fname:
                fake_file = fpath
            elif 'true' in fname:
                true_file = fpath
        
        return fake_file, true_file
    
    def _detect_any_csv_with_label(self, all_files: List[Path]) -> Optional[Path]:
        """Fallback: detect any CSV with a label column."""
        for f in all_files:
            try:
                df = pd.read_csv(f, nrows=100, encoding='utf-8', on_bad_lines='skip')
                cols_lower = {c.lower() for c in df.columns}
                if any(t in cols_lower for t in self.TARGET_NAMES):
                    return f
            except Exception:
                continue
        return None
    
    def _detect_split_column(self, df: pd.DataFrame) -> Optional[str]:
        """Detect split column (train/test/validation indicator)."""
        split_candidates = {'split', 'is_train', 'is_test', 'train_flag', 'test_flag', 'data_split', 'phase'}
        
        for col in df.columns:
            if col.lower() in split_candidates:
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) <= 5:
                    return col
        return None
    
    def _detect_target_column(self, df: pd.DataFrame) -> Optional[str]:
        """Detect target/label column."""
        cols_lower = {c.lower(): c for c in df.columns}
        
        for name in self.TARGET_NAMES:
            if name in cols_lower:
                return cols_lower[name]
        
        return None
    
    def _detect_text_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect text columns."""
        text_cols = []
        
        for col in df.columns:
            col_lower = col.lower()
            if any(t in col_lower for t in self.TEXT_COLUMNS):
                text_cols.append(col)
            elif df[col].dtype == object:
                sample = df[col].dropna().head(10)
                if len(sample) > 0:
                    avg_len = sample.astype(str).str.len().mean()
                    if avg_len > 50:
                        text_cols.append(col)
        
        return text_cols[:3]
    
    def _get_dataset_type(self, train_file, test_file, single_file, split_column) -> DatasetType:
        """Determine dataset type."""
        if train_file and test_file:
            return "fake_true_split" if 'fake' in str(train_file).lower() or 'true' in str(test_file).lower() else "train_test_split"
        elif single_file and split_column:
            return "single_csv"
        elif single_file:
            return "single_csv"
        elif train_file:
            return "train_test_split"
        return "unknown"
    
    def _load_sample(self, train_file, test_file, single_file) -> Optional[pd.DataFrame]:
        """Load a sample of the dataset."""
        source = train_file or test_file or single_file
        if source:
            return self._safe_read(source, nrows=100)
        return None
    
    def _safe_read(self, path: Path, nrows: int = None) -> Optional[pd.DataFrame]:
        """Safely read a CSV file."""
        for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
            try:
                return pd.read_csv(path, nrows=nrows, encoding=encoding, on_bad_lines='skip')
            except Exception:
                continue
        return None
    
    def _count_total_rows(self, train_file, test_file, validation_file, single_file) -> int:
        """Count total rows in the dataset."""
        total = 0
        for f in [train_file, test_file, validation_file, single_file]:
            if f:
                df = self._safe_read(f)
                if df is not None:
                    total += len(df)
        return total
    
    def save_registry(self, output_path: Path) -> None:
        """Save dataset registry to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        registry = {
            "total_datasets": len(self.datasets),
            "datasets": [d.to_dict() for d in self.datasets]
        }
        
        with open(output_path, 'w') as f:
            json.dump(registry, f, indent=2)
        
        logger.info(f"Saved dataset registry to: {output_path}")


def discover_datasets(root_path: str | Path, output_path: Optional[Path] = None) -> List[DatasetInfo]:
    """Discover datasets and optionally save registry."""
    discovery = DatasetDiscovery(root_path)
    datasets = discovery.discover()
    
    if output_path:
        discovery.save_registry(output_path)
    
    return datasets


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    dataset_path = r"C:\Users\JHASHANK\Downloads\ml_PROJ\dataset"
    output_path = r"C:\Users\JHASHANK\Downloads\ml_PROJ\outputs\datasets_registry.json"
    
    datasets = discover_datasets(dataset_path, output_path)
    
    print(f"\nDiscovered {len(datasets)} datasets:")
    for d in datasets:
        print(f"  - {d.name}: {d.dataset_type} ({d.n_rows} rows)")