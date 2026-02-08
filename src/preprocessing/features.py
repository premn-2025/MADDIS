"""
Feature engineering for drug discovery ML models

Creates domain-specific features for:
- Binding affinity prediction
- ADMET property prediction
- Toxicity prediction
- Drug-likeness scoring
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DrugFeatureEngineer:
    """Feature engineering for drug discovery tasks"""

    def __init__(self):
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, Crippen, Lipinski, QED
            self.Chem = Chem
            self.Descriptors = Descriptors
            self.Crippen = Crippen
            self.Lipinski = Lipinski
            self.QED = QED
            self.available = True
        except ImportError:
            logger.error("RDKit not available for feature engineering")
            self.available = False

    def calculate_admet_features(self, mol) -> Dict:
        """Calculate ADMET-relevant molecular descriptors"""
        if not self.available:
            return {}

        features = {}

        try:
            features['molecular_weight'] = self.Descriptors.MolWt(mol)
            features['logp'] = self.Descriptors.MolLogP(mol)
            features['tpsa'] = self.Descriptors.TPSA(mol)
            features['hbd'] = self.Descriptors.NumHDonors(mol)
            features['hba'] = self.Descriptors.NumHAcceptors(mol)
            features['rotatable_bonds'] = self.Descriptors.NumRotatableBonds(mol)
            features['rigid_bonds'] = mol.GetNumBonds() - features['rotatable_bonds']
            features['ring_count'] = self.Descriptors.RingCount(mol)
            features['aromatic_rings'] = self.Descriptors.NumAromaticRings(mol)
            features['saturated_rings'] = self.Descriptors.NumSaturatedRings(mol)
            features['formal_charge'] = self.Chem.GetFormalCharge(mol)
            features['lipinski_violations'] = self._count_lipinski_violations(mol)
            features['qed_score'] = self.QED.qed(mol)
            features['heavy_atoms'] = mol.GetNumHeavyAtoms()
            features['heteroatoms'] = self._count_heteroatoms(mol)
            features['molar_refractivity'] = self.Crippen.MolMR(mol)
            features['fraction_csp3'] = self._calculate_fraction_sp3(mol)

        except Exception as e:
            logger.warning(f"Error calculating ADMET features: {e}")

        return features

    def calculate_binding_features(self, mol) -> Dict:
        """Calculate features relevant for binding affinity prediction"""
        if not self.available:
            return {}

        features = {}

        try:
            features['molecular_weight'] = self.Descriptors.MolWt(mol)
            features['heavy_atoms'] = mol.GetNumHeavyAtoms()
            features['formal_charge'] = self.Chem.GetFormalCharge(mol)
            features['polar_surface_area'] = self.Descriptors.TPSA(mol)
            features['logp'] = self.Descriptors.MolLogP(mol)
            features['aromatic_proportion'] = self.Descriptors.NumAromaticRings(mol) / max(self.Descriptors.RingCount(mol), 1)
            features['fraction_csp3'] = self._calculate_fraction_sp3(mol)
            features['rotatable_bonds'] = self.Descriptors.NumRotatableBonds(mol)
            features['flexibility_ratio'] = features['rotatable_bonds'] / max(mol.GetNumBonds(), 1)
            features['hbd'] = self.Descriptors.NumHDonors(mol)
            features['hba'] = self.Descriptors.NumHAcceptors(mol)
            features['hb_ratio'] = (features['hbd'] + features['hba']) / features['heavy_atoms']
            features['ring_count'] = self.Descriptors.RingCount(mol)
            features['aromatic_rings'] = self.Descriptors.NumAromaticRings(mol)

        except Exception as e:
            logger.warning(f"Error calculating binding features: {e}")

        return features

    def calculate_toxicity_features(self, mol) -> Dict:
        """Calculate features relevant for toxicity prediction"""
        if not self.available:
            return {}

        features = {}

        try:
            features['reactive_groups'] = self._count_reactive_groups(mol)
            features['aromatic_rings'] = self.Descriptors.NumAromaticRings(mol)
            features['heterocycles'] = self._count_heterocycles(mol)
            features['molecular_weight'] = self.Descriptors.MolWt(mol)
            features['logp'] = self.Descriptors.MolLogP(mol)
            features['formal_charge'] = self.Chem.GetFormalCharge(mol)

        except Exception as e:
            logger.warning(f"Error calculating toxicity features: {e}")

        return features

    def _count_lipinski_violations(self, mol) -> int:
        """Count Lipinski Rule of Five violations"""
        violations = 0
        if self.Descriptors.MolWt(mol) > 500:
            violations += 1
        if self.Descriptors.MolLogP(mol) > 5:
            violations += 1
        if self.Descriptors.NumHDonors(mol) > 5:
            violations += 1
        if self.Descriptors.NumHAcceptors(mol) > 10:
            violations += 1
        return violations

    def _count_heteroatoms(self, mol) -> int:
        """Count non-carbon heavy atoms"""
        count = 0
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() not in [1, 6]:
                count += 1
        return count

    def _count_reactive_groups(self, mol) -> int:
        """Count potentially reactive functional groups"""
        reactive_patterns = [
            '[CX3](=O)[OX2H1]',
            '[CX3](=O)[CX4]',
            '[CX3H1](=O)',
            '[NX3][CX3](=[OX1])',
            '[SX2][CX3](=[OX1])',
            '[CX3](=[OX1])[OX2][CX4]',
        ]

        count = 0
        for pattern in reactive_patterns:
            try:
                patt = self.Chem.MolFromSmarts(pattern)
                if patt:
                    matches = mol.GetSubstructMatches(patt)
                    count += len(matches)
            except Exception:
                continue

        return count

    def _count_heterocycles(self, mol) -> int:
        """Count heterocyclic rings"""
        ring_info = mol.GetRingInfo()
        heterocycle_count = 0

        for ring in ring_info.AtomRings():
            for atom_idx in ring:
                atom = mol.GetAtomWithIdx(atom_idx)
                if atom.GetAtomicNum() not in [6, 1]:
                    heterocycle_count += 1
                    break

        return heterocycle_count

    def _calculate_fraction_sp3(self, mol) -> float:
        """Calculate fraction of SP3 carbons"""
        try:
            sp3_carbons = 0
            total_carbons = 0

            for atom in mol.GetAtoms():
                if atom.GetSymbol() == 'C':
                    total_carbons += 1
                    if atom.GetHybridization() == self.Chem.HybridizationType.SP3:
                        sp3_carbons += 1

            if total_carbons == 0:
                return 0.0

            return sp3_carbons / total_carbons
        except Exception:
            return 0.0


class ProteinFeatureEngineer:
    """Feature engineering for protein structures"""

    def __init__(self):
        self.amino_acids = {
            'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
            'E': 'GLU', 'Q': 'GLN', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
            'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
            'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
        }

        self.aa_properties = {
            'hydrophobic': ['A', 'I', 'L', 'M', 'F', 'W', 'Y', 'V'],
            'polar': ['N', 'C', 'Q', 'S', 'T'],
            'charged': ['R', 'H', 'K', 'D', 'E'],
            'aromatic': ['F', 'W', 'Y', 'H'],
            'small': ['A', 'C', 'D', 'G', 'N', 'P', 'S', 'T', 'V']
        }

    def calculate_sequence_features(self, sequence: str) -> Dict:
        """Calculate features from protein sequence"""
        features = {}

        features['length'] = len(sequence)
        for aa in self.amino_acids:
            features[f'freq_{aa}'] = sequence.count(aa) / len(sequence)

        for prop, aas in self.aa_properties.items():
            count = sum(sequence.count(aa) for aa in aas)
            features[f'{prop}_fraction'] = count / len(sequence)

        features['charge'] = self._calculate_charge(sequence)
        features['isoelectric_point'] = self._estimate_pi(sequence)
        features['hydropathy'] = self._calculate_hydropathy(sequence)

        return features

    def _calculate_charge(self, sequence: str, pH: float = 7.0) -> float:
        """Calculate net charge at given pH"""
        positive = sequence.count('R') + sequence.count('K') + sequence.count('H')
        negative = sequence.count('D') + sequence.count('E')
        return positive - negative

    def _estimate_pi(self, sequence: str) -> float:
        """Estimate isoelectric point"""
        positive = sequence.count('R') + sequence.count('K') + sequence.count('H')
        negative = sequence.count('D') + sequence.count('E')

        if positive > negative:
            return 8.5
        elif negative > positive:
            return 5.5
        else:
            return 7.0

    def _calculate_hydropathy(self, sequence: str) -> float:
        """Calculate average hydropathy index"""
        hydropathy_scale = {
            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
            'E': -3.5, 'Q': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
            'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
            'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
        }

        total_hydropathy = sum(hydropathy_scale.get(aa, 0) for aa in sequence)
        return total_hydropathy / len(sequence)


class InteractionFeatureEngineer:
    """Feature engineering for protein-ligand interactions"""

    def calculate_interaction_features(self, ligand_features: Dict, protein_features: Dict) -> Dict:
        """Calculate features describing protein-ligand interactions"""
        features = {}

        features['size_ratio'] = ligand_features.get('molecular_weight', 0) / 500.0

        ligand_logp = ligand_features.get('logp', 0)
        protein_hydrophobic = protein_features.get('hydrophobic_fraction', 0)
        features['hydrophobic_complementarity'] = ligand_logp * protein_hydrophobic

        ligand_hb = (ligand_features.get('hbd', 0) + ligand_features.get('hba', 0))
        protein_polar = protein_features.get('polar_fraction', 0)
        features['hb_potential'] = ligand_hb * protein_polar

        ligand_charge = ligand_features.get('formal_charge', 0)
        protein_charge = protein_features.get('charge', 0)
        features['electrostatic_complementarity'] = -ligand_charge * protein_charge

        ligand_flexibility = ligand_features.get('rotatable_bonds', 0)
        features['shape_complementarity'] = 1.0 / (1.0 + ligand_flexibility * 0.1)

        return features


if __name__ == "__main__":
    engineer = DrugFeatureEngineer()
    
    if engineer.available:
        from rdkit import Chem
        mol = Chem.MolFromSmiles("CCO")
        
        admet_features = engineer.calculate_admet_features(mol)
        binding_features = engineer.calculate_binding_features(mol)
        
        print("ADMET Features:", admet_features)
        print("Binding Features:", binding_features)
