#!/usr/bin/env python3
"""
Molecular Dataset Builder

Production-ready system for downloading and processing molecular data from ChEMBL/ZINC
for fine-tuning AI models. Generates chemically accurate 2D/3D representations.
"""

import os
import sys
import csv
import json
import logging
import requests
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem, Descriptors
    HAS_RDKIT = True
except ImportError:
    print("RDKit not found. Install with: conda install -c conda-forge rdkit")
    HAS_RDKIT = False


@dataclass
class DatasetConfig:
    """Configuration for dataset generation"""
    source: str = "chembl"
    total_molecules: int = 60000
    output_dir: str = "drug_dataset"
    generate_2d_images: bool = True
    generate_3d_conformers: bool = True
    image_size: Tuple[int, int] = (300, 300)
    filter_drug_like: bool = True
    min_mw: float = 150.0
    max_mw: float = 800.0
    min_logp: float = -3.0
    max_logp: float = 8.0
    batch_size: int = 1000
    max_retries: int = 3
    delay_between_requests: float = 0.1


class MolecularDatasetBuilder:
    """Professional molecular dataset builder for AI training"""

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.setup_logging()
        self.setup_directories()

        if not HAS_RDKIT:
            raise ImportError("RDKit is required. Install with: conda install -c conda-forge rdkit")

    def setup_logging(self):
        """Setup comprehensive logging"""
        os.makedirs(f"{self.config.output_dir}/logs", exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{self.config.output_dir}/logs/dataset_builder.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_directories(self):
        """Create output directory structure"""
        dirs = [
            self.config.output_dir,
            f"{self.config.output_dir}/images_2d",
            f"{self.config.output_dir}/conformers_3d",
            f"{self.config.output_dir}/metadata",
            f"{self.config.output_dir}/logs"
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)

    def download_chembl_molecules(self, limit: int = 1000) -> List[Dict]:
        """Download molecules from ChEMBL"""
        self.logger.info(f"Downloading {limit} molecules from ChEMBL...")

        molecules = []
        offset = 0
        batch_size = min(1000, limit)

        while len(molecules) < limit:
            url = f"https://www.ebi.ac.uk/chembl/api/data/molecule.json?limit={batch_size}&offset={offset}"

            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                data = response.json()

                for mol in data.get('molecules', []):
                    if mol.get('molecule_structures') and mol['molecule_structures'].get('canonical_smiles'):
                        smiles = mol['molecule_structures']['canonical_smiles']
                        
                        if self._is_valid_smiles(smiles):
                            molecules.append({
                                'smiles': smiles,
                                'chembl_id': mol.get('molecule_chembl_id', ''),
                                'pref_name': mol.get('pref_name', '')
                            })

                            if len(molecules) >= limit:
                                break

                offset += batch_size
                time.sleep(self.config.delay_between_requests)

            except Exception as e:
                self.logger.warning(f"ChEMBL download error: {e}")
                break

        self.logger.info(f"Downloaded {len(molecules)} molecules from ChEMBL")
        return molecules

    def _is_valid_smiles(self, smiles: str) -> bool:
        """Validate SMILES string"""
        if not smiles or len(smiles) < 2:
            return False

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False

            mw = Descriptors.MolWt(mol)
            if mw < self.config.min_mw or mw > self.config.max_mw:
                return False

            logp = Descriptors.MolLogP(mol)
            if logp < self.config.min_logp or logp > self.config.max_logp:
                return False

            return True

        except Exception:
            return False

    def generate_2d_image(self, smiles: str, output_path: str) -> bool:
        """Generate 2D molecular image"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False

            img = Draw.MolToImage(mol, size=self.config.image_size)
            img.save(output_path)
            return True

        except Exception as e:
            self.logger.warning(f"2D image generation failed: {e}")
            return False

    def generate_3d_conformer(self, smiles: str, output_path: str) -> bool:
        """Generate 3D conformer and save as SDF"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False

            mol = Chem.AddHs(mol)
            
            result = AllChem.EmbedMolecule(mol, randomSeed=42)
            if result != 0:
                return False

            AllChem.MMFFOptimizeMolecule(mol)

            writer = Chem.SDWriter(output_path)
            writer.write(mol)
            writer.close()

            return True

        except Exception as e:
            self.logger.warning(f"3D conformer generation failed: {e}")
            return False

    def calculate_properties(self, smiles: str) -> Dict:
        """Calculate molecular properties"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {}

            return {
                'molecular_weight': Descriptors.MolWt(mol),
                'logp': Descriptors.MolLogP(mol),
                'tpsa': Descriptors.TPSA(mol),
                'hbd': Descriptors.NumHDonors(mol),
                'hba': Descriptors.NumHAcceptors(mol),
                'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'ring_count': Descriptors.RingCount(mol),
                'heavy_atoms': mol.GetNumHeavyAtoms()
            }

        except Exception:
            return {}

    def build_dataset(self, molecules: List[Dict]) -> pd.DataFrame:
        """Build complete dataset with images and properties"""
        self.logger.info(f"Building dataset for {len(molecules)} molecules...")

        records = []

        for i, mol_data in enumerate(molecules):
            smiles = mol_data['smiles']
            mol_id = mol_data.get('chembl_id', f'mol_{i:06d}')

            record = {
                'id': mol_id,
                'smiles': smiles,
                'name': mol_data.get('pref_name', '')
            }

            if self.config.generate_2d_images:
                img_path = f"{self.config.output_dir}/images_2d/{mol_id}.png"
                if self.generate_2d_image(smiles, img_path):
                    record['image_2d'] = img_path

            if self.config.generate_3d_conformers:
                sdf_path = f"{self.config.output_dir}/conformers_3d/{mol_id}.sdf"
                if self.generate_3d_conformer(smiles, sdf_path):
                    record['conformer_3d'] = sdf_path

            properties = self.calculate_properties(smiles)
            record.update(properties)

            records.append(record)

            if (i + 1) % 100 == 0:
                self.logger.info(f"Processed {i + 1}/{len(molecules)} molecules")

        df = pd.DataFrame(records)
        df.to_csv(f"{self.config.output_dir}/metadata/dataset.csv", index=False)

        self.logger.info(f"Dataset built with {len(df)} molecules")
        return df

    def run(self, limit: int = 1000) -> pd.DataFrame:
        """Run the complete dataset building pipeline"""
        self.logger.info("Starting dataset building pipeline...")

        molecules = self.download_chembl_molecules(limit)

        if not molecules:
            self.logger.error("No molecules downloaded")
            return pd.DataFrame()

        df = self.build_dataset(molecules)

        self.logger.info("Dataset building completed")
        return df


if __name__ == "__main__":
    config = DatasetConfig(
        total_molecules=100,
        output_dir="test_dataset"
    )

    builder = MolecularDatasetBuilder(config)
    df = builder.run(limit=100)
    print(f"Built dataset with {len(df)} molecules")
