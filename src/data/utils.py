"""
Data utilities and helper functions for drug discovery pipeline
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path
import pickle
import json

logger = logging.getLogger(__name__)


class DataUtils:
    """Utility functions for data processing and management"""

    @staticmethod
    def validate_smiles(smiles: str) -> bool:
        """Validate SMILES string using RDKit"""
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except ImportError:
            logger.warning("RDKit not available for SMILES validation")
            return len(smiles) > 0

    @staticmethod
    def clean_activity_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize bioactivity data"""
        df_clean = df.copy()

        # Remove invalid SMILES
        if 'smiles' in df_clean.columns:
            valid_smiles = df_clean['smiles'].apply(DataUtils.validate_smiles)
            df_clean = df_clean[valid_smiles]
            logger.info(f"Filtered to {len(df_clean)} molecules with valid SMILES")

        # Standardize activity values
        if 'activity_value' in df_clean.columns:
            df_clean = df_clean.dropna(subset=['activity_value'])

            # Convert to pIC50/pKi (negative log10 of molar concentration)
            if 'activity_type' in df_clean.columns:
                df_clean['pActivity'] = df_clean.apply(
                    lambda row: DataUtils.convert_to_pactivity(
                        row['activity_value'],
                        row.get('activity_type', 'IC50')
                    ), axis=1
                )

        return df_clean

    @staticmethod
    def convert_to_pactivity(value: float, activity_type: str) -> float:
        """Convert activity values to pIC50/pKi scale"""
        try:
            # Assume values are in nM, convert to M then take -log10
            if activity_type.upper() in ['IC50', 'KI', 'KD', 'EC50']:
                # Convert nM to M
                molar_value = value * 1e-9
                return -np.log10(molar_value)
            else:
                return value
        except (ValueError, TypeError):
            return np.nan

    @staticmethod
    def filter_drug_like(df: pd.DataFrame) -> pd.DataFrame:
        """Filter molecules by drug-likeness criteria (Lipinski's Rule of Five)"""
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors

            def is_drug_like(smiles: str) -> bool:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return False

                # Lipinski's Rule of Five
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                hbd = Descriptors.NumHDonors(mol)
                hba = Descriptors.NumHAcceptors(mol)

                return (mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10)

            if 'smiles' in df.columns:
                drug_like = df['smiles'].apply(is_drug_like)
                df_filtered = df[drug_like]
                logger.info(f"Filtered to {len(df_filtered)} drug-like molecules")
                return df_filtered

        except ImportError:
            logger.warning("RDKit not available for drug-likeness filtering")

        return df

    @staticmethod
    def split_data(df: pd.DataFrame,
                   test_size: float = 0.2,
                   val_size: float = 0.1,
                   random_state: int = 42) -> Dict[str, pd.DataFrame]:
        """Split data into train/validation/test sets"""
        from sklearn.model_selection import train_test_split

        # First split: train+val vs test
        train_val, test = train_test_split(
            df, test_size=test_size, random_state=random_state
        )

        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val, test_size=val_size_adjusted, random_state=random_state)

        splits = {
            'train': train,
            'val': val,
            'test': test
        }

        logger.info(f"Data split - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        return splits


class DataCache:
    """Simple caching system for data persistence"""

    def __init__(self, cache_dir: str = "./data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def save(self, data: any, key: str, format: str = 'pickle') -> None:
        """Save data to cache"""
        if format == 'pickle':
            filepath = self.cache_dir / f"{key}.pkl"
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        elif format == 'csv' and isinstance(data, pd.DataFrame):
            filepath = self.cache_dir / f"{key}.csv"
            data.to_csv(filepath, index=False)
        elif format == 'json':
            filepath = self.cache_dir / f"{key}.json"
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Cached data: {filepath}")

    def load(self, key: str, format: str = 'pickle') -> any:
        """Load data from cache"""
        if format == 'pickle':
            filepath = self.cache_dir / f"{key}.pkl"
        elif format == 'csv':
            filepath = self.cache_dir / f"{key}.csv"
        elif format == 'json':
            filepath = self.cache_dir / f"{key}.json"
        else:
            raise ValueError(f"Unsupported format: {format}")

        if not filepath.exists():
            logger.warning(f"Cache file not found: {filepath}")
            return None

        try:
            if format == 'pickle':
                with open(filepath, 'rb') as f:
                    return pickle.load(f)
            elif format == 'csv':
                return pd.read_csv(filepath)
            elif format == 'json':
                with open(filepath, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading cache file {filepath}: {e}")
            return None

    def exists(self, key: str, format: str = 'pickle') -> bool:
        """Check if cached data exists"""
        if format == 'pickle':
            filepath = self.cache_dir / f"{key}.pkl"
        elif format == 'csv':
            filepath = self.cache_dir / f"{key}.csv"
        elif format == 'json':
            filepath = self.cache_dir / f"{key}.json"
        else:
            return False

        return filepath.exists()


class DatasetBuilder:
    """Build ML-ready datasets for drug discovery tasks"""

    def __init__(self, cache_dir: str = "./data/cache"):
        self.cache = DataCache(cache_dir)
        self.utils = DataUtils()

    def build_activity_dataset(self, target_name: str, molecules_df: pd.DataFrame) -> Dict:
        """Build activity prediction dataset"""
        # Clean and validate data
        df_clean = self.utils.clean_activity_data(molecules_df)
        df_clean = self.utils.filter_drug_like(df_clean)

        # Create binary labels (active/inactive)
        if 'pActivity' in df_clean.columns:
            # Threshold for activity (e.g., pIC50 > 6 means IC50 < 1μM)
            df_clean['active'] = (df_clean['pActivity'] > 6).astype(int)

        # Split data
        splits = self.utils.split_data(df_clean)

        # Create final dataset
        dataset = {
            'target': target_name,
            'data': splits,
            'metadata': {
                'total_molecules': len(df_clean),
                'active_molecules': df_clean['active'].sum() if 'active' in df_clean.columns else 0,
                'features': list(df_clean.columns),
                'created': pd.Timestamp.now().isoformat()
            }
        }

        # Cache the dataset
        cache_key = f"dataset_{target_name.lower().replace(' ', '_')}"
        self.cache.save(dataset, cache_key)

        logger.info(f"Built activity dataset for {target_name}: {len(df_clean)} molecules")
        return dataset

    def load_dataset(self, target_name: str) -> Optional[Dict]:
        """Load cached dataset"""
        cache_key = f"dataset_{target_name.lower().replace(' ', '_')}"
        return self.cache.load(cache_key)


# Example usage
if __name__ == "__main__":
    # Example of using data utilities
    from collectors import DataManager

    # Collect sample data
    manager = DataManager()
    target_data = manager.collect_target_data("EGFR", limit=50)

    if not target_data.empty:
        # Build dataset
        builder = DatasetBuilder()
        dataset = builder.build_activity_dataset("EGFR", target_data)

        print(f"Dataset built with {dataset['metadata']['total_molecules']} molecules")
        print(f"Active molecules: {dataset['metadata']['active_molecules']}")
    else:
        print("No data collected for example")
