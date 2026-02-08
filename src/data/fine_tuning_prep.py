#!/usr/bin/env python3
"""
Fine-Tuning Data Preparation for Drug Discovery Models

Prepares molecular datasets for fine-tuning various AI architectures:
- Vision models (ViT, ResNet, EfficientNet)
- Multimodal models (CLIP, BLIP)
- Graph Neural Networks
- Molecular transformers
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    HAS_CHEMISTRY = True
except ImportError:
    print("Chemistry packages not found. Install rdkit")
    HAS_CHEMISTRY = False

logger = logging.getLogger(__name__)


@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning data preparation"""
    dataset_dir: str = "drug_dataset"
    output_dir: str = "fine_tuning_data"
    prepare_vision: bool = True
    prepare_multimodal: bool = True
    prepare_graph: bool = True
    prepare_transformer: bool = True
    image_size: int = 224
    augmentation_strength: str = "medium"
    property_targets: List[str] = None
    text_descriptions: bool = True
    batch_size: int = 32
    num_workers: int = 4
    device: str = "auto"

    def __post_init__(self):
        if self.property_targets is None:
            self.property_targets = ["molecular_weight", "logp", "hba", "hbd", "tpsa"]


class MolecularImageDataset(Dataset):
    """PyTorch Dataset for molecular images"""

    def __init__(self, image_dir: str, metadata_file: str,
                 transform: Optional[transforms.Compose] = None,
                 target_properties: List[str] = None,
                 indices: Optional[List[int]] = None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.target_properties = target_properties or []

        self.metadata = pd.read_csv(metadata_file)

        if indices is not None:
            self.metadata = self.metadata.iloc[indices].reset_index(drop=True)

        self.valid_indices = []
        for idx, row in self.metadata.iterrows():
            img_path = self.image_dir / f"{row.get('id', f'mol_{idx}')}.png"
            if img_path.exists():
                self.valid_indices.append(idx)

        logger.info(f"Loaded {len(self.valid_indices)} valid images from {len(self.metadata)} entries")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        row = self.metadata.iloc[actual_idx]

        mol_id = row.get('id', f'mol_{actual_idx}')
        img_path = self.image_dir / f"{mol_id}.png"

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Failed to load image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color='white')

        if self.transform:
            image = self.transform(image)

        targets = {}
        for prop in self.target_properties:
            if prop in row:
                targets[prop] = float(row[prop])

        return {
            'image': image,
            'smiles': row.get('smiles', ''),
            'mol_id': mol_id,
            'targets': targets
        }


class MultimodalMolecularDataset(Dataset):
    """Dataset for multimodal learning with images, SMILES, and properties"""

    def __init__(self, image_dir: str, metadata_file: str,
                 transform: Optional[transforms.Compose] = None,
                 tokenizer=None, max_text_length: int = 128):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length

        self.metadata = pd.read_csv(metadata_file)
        self.valid_indices = []

        for idx, row in self.metadata.iterrows():
            img_path = self.image_dir / f"{row.get('id', f'mol_{idx}')}.png"
            if img_path.exists() and row.get('smiles'):
                self.valid_indices.append(idx)

        logger.info(f"Loaded {len(self.valid_indices)} multimodal samples")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        row = self.metadata.iloc[actual_idx]

        mol_id = row.get('id', f'mol_{actual_idx}')
        img_path = self.image_dir / f"{mol_id}.png"

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), color='white')

        if self.transform:
            image = self.transform(image)

        smiles = row.get('smiles', '')
        text_description = self._generate_description(row)

        return {
            'image': image,
            'smiles': smiles,
            'text': text_description,
            'mol_id': mol_id
        }

    def _generate_description(self, row) -> str:
        """Generate text description from molecular properties"""
        parts = []

        if 'pref_name' in row and row['pref_name']:
            parts.append(f"This molecule is {row['pref_name']}.")

        if 'molecular_weight' in row:
            parts.append(f"It has a molecular weight of {row['molecular_weight']:.1f} Da.")

        if 'logp' in row:
            parts.append(f"The calculated LogP is {row['logp']:.2f}.")

        if not parts:
            parts.append(f"This is a molecular structure with SMILES: {row.get('smiles', 'unknown')}")

        return ' '.join(parts)


class FineTuningDataPreparer:
    """Main class for preparing fine-tuning datasets"""

    def __init__(self, config: FineTuningConfig):
        self.config = config
        self.setup_directories()
        self.setup_device()
        self.setup_transforms()

    def setup_directories(self):
        """Create output directories"""
        dirs = [
            self.config.output_dir,
            f"{self.config.output_dir}/vision",
            f"{self.config.output_dir}/multimodal",
            f"{self.config.output_dir}/graph",
            f"{self.config.output_dir}/transformer"
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)

    def setup_device(self):
        """Setup computing device"""
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
        logger.info(f"Using device: {self.device}")

    def setup_transforms(self):
        """Setup image transformations"""
        strength = self.config.augmentation_strength

        if strength == "light":
            self.train_transform = transforms.Compose([
                transforms.Resize((self.config.image_size, self.config.image_size)),
                transforms.RandomHorizontalFlip(0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif strength == "medium":
            self.train_transform = transforms.Compose([
                transforms.Resize((self.config.image_size, self.config.image_size)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.train_transform = transforms.Compose([
                transforms.Resize((self.config.image_size, self.config.image_size)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(30),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        self.val_transform = transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def prepare_vision_data(self, metadata_file: str, image_dir: str) -> Dict:
        """Prepare data for vision model fine-tuning"""
        logger.info("Preparing vision model data...")

        metadata = pd.read_csv(metadata_file)
        n_samples = len(metadata)

        indices = np.random.permutation(n_samples)
        train_split = int(0.8 * n_samples)
        val_split = int(0.9 * n_samples)

        train_indices = indices[:train_split].tolist()
        val_indices = indices[train_split:val_split].tolist()
        test_indices = indices[val_split:].tolist()

        train_dataset = MolecularImageDataset(
            image_dir, metadata_file,
            transform=self.train_transform,
            target_properties=self.config.property_targets,
            indices=train_indices
        )

        val_dataset = MolecularImageDataset(
            image_dir, metadata_file,
            transform=self.val_transform,
            target_properties=self.config.property_targets,
            indices=val_indices
        )

        test_dataset = MolecularImageDataset(
            image_dir, metadata_file,
            transform=self.val_transform,
            target_properties=self.config.property_targets,
            indices=test_indices
        )

        split_info = {
            'train_size': len(train_dataset),
            'val_size': len(val_dataset),
            'test_size': len(test_dataset),
            'train_indices': train_indices,
            'val_indices': val_indices,
            'test_indices': test_indices
        }

        with open(f"{self.config.output_dir}/vision/split_info.json", 'w') as f:
            json.dump(split_info, f, indent=2)

        logger.info(f"Vision data prepared: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

        return {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset,
            'split_info': split_info
        }

    def prepare_multimodal_data(self, metadata_file: str, image_dir: str) -> Dict:
        """Prepare data for multimodal learning"""
        logger.info("Preparing multimodal data...")

        dataset = MultimodalMolecularDataset(
            image_dir, metadata_file,
            transform=self.train_transform
        )

        logger.info(f"Multimodal data prepared: {len(dataset)} samples")

        return {'dataset': dataset}

    def create_dataloaders(self, datasets: Dict) -> Dict:
        """Create DataLoaders from datasets"""
        loaders = {}

        if 'train' in datasets:
            loaders['train'] = DataLoader(
                datasets['train'],
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                pin_memory=True if self.device.type == 'cuda' else False
            )

        if 'val' in datasets:
            loaders['val'] = DataLoader(
                datasets['val'],
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers
            )

        if 'test' in datasets:
            loaders['test'] = DataLoader(
                datasets['test'],
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers
            )

        return loaders


if __name__ == "__main__":
    config = FineTuningConfig(
        dataset_dir="drug_dataset",
        output_dir="fine_tuning_output",
        image_size=224
    )

    preparer = FineTuningDataPreparer(config)
    print("Fine-tuning data preparer initialized successfully")
