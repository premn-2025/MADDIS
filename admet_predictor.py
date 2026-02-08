#!/usr/bin/env python3
"""
ADMET PROPERTY PREDICTOR - GNN-Based Property Prediction

This module provides ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity)
prediction using Graph Neural Networks for drug discovery.

Features:
- GNN architecture for molecular property prediction
- Multi-task learning for simultaneous ADMET prediction
- Support for training on ToxCast/Tox21 datasets
- Easy integration with RL generator
"""

import os
import json
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

logger = logging.getLogger(__name__)


@dataclass
class ADMETResult:
    """ADMET prediction result for a molecule"""
    smiles: str
    absorption: float        # 0-1, higher = better
    distribution: float      # 0-1, higher = better
    metabolism: float        # 0-1, higher = more stable
    excretion: float         # 0-1, higher = easier clearance
    toxicity_risk: float     # 0-1, higher = MORE TOXIC (bad)
    herg_risk: float         # 0-1, cardiac toxicity risk
    hepatotoxicity: float    # 0-1, liver toxicity risk
    bbb_permeability: float  # 0-1, blood-brain barrier crossing
    solubility: float        # 0-1, aqueous solubility
    confidence: str          # high, medium, low
    overall_score: float     # Combined ADMET score (higher = better drug candidate)


class MolecularFeaturizer:
    """Convert molecules to graph features for GNN"""
    
    # Atom features
    ATOM_FEATURES = {
        'atomic_num': list(range(1, 119)),
        'chirality': ['CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW', 'CHI_OTHER'],
        'degree': [0, 1, 2, 3, 4, 5, 6],
        'formal_charge': [-2, -1, 0, 1, 2],
        'num_hs': [0, 1, 2, 3, 4],
        'hybridization': ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2'],
        'aromatic': [False, True],
        'in_ring': [False, True],
    }
    
    def __init__(self):
        self.atom_feature_dim = self._calc_atom_feature_dim()
    
    def _calc_atom_feature_dim(self) -> int:
        """Calculate total atom feature dimension"""
        dim = 0
        for key, values in self.ATOM_FEATURES.items():
            dim += len(values)
        return dim
    
    def _one_hot(self, value: Any, choices: List) -> List[int]:
        """Create one-hot encoding"""
        encoding = [0] * len(choices)
        if value in choices:
            encoding[choices.index(value)] = 1
        else:
            encoding[0] = 1  # Default to first
        return encoding
    
    def get_atom_features(self, atom) -> List[float]:
        """Extract features for a single atom"""
        features = []
        features += self._one_hot(atom.GetAtomicNum(), self.ATOM_FEATURES['atomic_num'])
        features += self._one_hot(str(atom.GetChiralTag()), self.ATOM_FEATURES['chirality'])
        features += self._one_hot(atom.GetDegree(), self.ATOM_FEATURES['degree'])
        features += self._one_hot(atom.GetFormalCharge(), self.ATOM_FEATURES['formal_charge'])
        features += self._one_hot(atom.GetTotalNumHs(), self.ATOM_FEATURES['num_hs'])
        features += self._one_hot(str(atom.GetHybridization()), self.ATOM_FEATURES['hybridization'])
        features += self._one_hot(atom.GetIsAromatic(), self.ATOM_FEATURES['aromatic'])
        features += self._one_hot(atom.IsInRing(), self.ATOM_FEATURES['in_ring'])
        return features
    
    def mol_to_graph(self, mol) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert RDKit mol to graph tensors"""
        if mol is None:
            raise ValueError("Invalid molecule")
        
        # Atom features
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append(self.get_atom_features(atom))
        
        x = torch.tensor(atom_features, dtype=torch.float)
        
        # Edge indices (bonds)
        edge_indices = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_indices.extend([[i, j], [j, i]])  # Undirected
        
        if len(edge_indices) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).T
        
        return x, edge_index
    
    def smiles_to_graph(self, smiles: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Convert SMILES to graph tensors"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return self.mol_to_graph(mol)


class GNNLayer(nn.Module):
    """Graph Neural Network layer with message passing"""
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Message passing forward"""
        # Simple aggregation: mean of neighbors
        if edge_index.shape[1] > 0:
            row, col = edge_index
            deg = torch.bincount(row, minlength=x.size(0)).float().clamp(min=1)
            
            # Aggregate neighbor features
            agg = torch.zeros_like(x)
            agg.index_add_(0, row, x[col])
            agg = agg / deg.view(-1, 1)
            
            # Combine self and neighbor
            x = x + agg
        
        x = self.linear(x)
        if x.size(0) > 1:
            x = self.bn(x)
        return F.relu(x)


class ADMETPredictor(nn.Module):
    """
    Graph Neural Network for ADMET Property Prediction
    
    Multi-task learning model that predicts multiple ADMET properties
    from molecular structure.
    """
    
    def __init__(
        self,
        atom_feature_dim: int = 140,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_tasks: int = 9,  # absorption, distribution, metabolism, excretion, toxicity, hERG, hepato, BBB, solubility
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.featurizer = MolecularFeaturizer()
        self.atom_feature_dim = self.featurizer.atom_feature_dim
        
        # Input projection
        self.input_proj = nn.Linear(self.atom_feature_dim, hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            GNNLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # Graph-level readout
        self.pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multi-task output heads
        self.task_heads = nn.ModuleDict({
            'absorption': nn.Linear(hidden_dim, 1),
            'distribution': nn.Linear(hidden_dim, 1),
            'metabolism': nn.Linear(hidden_dim, 1),
            'excretion': nn.Linear(hidden_dim, 1),
            'toxicity': nn.Linear(hidden_dim, 1),
            'herg': nn.Linear(hidden_dim, 1),
            'hepatotoxicity': nn.Linear(hidden_dim, 1),
            'bbb': nn.Linear(hidden_dim, 1),
            'solubility': nn.Linear(hidden_dim, 1),
        })
        
        self.dropout = nn.Dropout(dropout)
        
        # Model state
        self.trained = False
        self.device = torch.device('cpu')
        
        logger.info(f"ADMET Predictor initialized: {num_layers} GNN layers, {hidden_dim} hidden dim")
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for a single molecule graph"""
        # Input projection
        h = self.input_proj(x)
        h = F.relu(h)
        
        # GNN message passing
        for gnn in self.gnn_layers:
            h = gnn(h, edge_index)
            h = self.dropout(h)
        
        # Global pooling (mean)
        graph_rep = h.mean(dim=0, keepdim=True)
        graph_rep = self.pool(graph_rep)
        
        # Multi-task predictions
        predictions = {}
        for task_name, head in self.task_heads.items():
            predictions[task_name] = torch.sigmoid(head(graph_rep))  # 0-1 output
        
        return predictions
    
    def predict_smiles(self, smiles: str) -> Optional[Dict[str, float]]:
        """Predict ADMET properties from SMILES"""
        graph = self.featurizer.smiles_to_graph(smiles)
        if graph is None:
            return None
        
        x, edge_index = graph
        x, edge_index = x.to(self.device), edge_index.to(self.device)
        
        self.eval()
        with torch.no_grad():
            preds = self.forward(x, edge_index)
        
        return {k: v.item() for k, v in preds.items()}


class ADMETAgent:
    """
    High-level ADMET Prediction Agent
    
    Provides easy-to-use ADMET predictions with rule-based fallbacks
    when GNN is not trained.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize ADMET agent with optional pre-trained model"""
        self.model = ADMETPredictor()
        self.model_loaded = False
        
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            logger.info("Using rule-based ADMET predictions (model not trained)")
    
    def _load_model(self, path: str):
        """Load pre-trained model"""
        try:
            self.model.load_state_dict(torch.load(path, map_location='cpu'))
            self.model.eval()
            self.model_loaded = True
            logger.info(f"Loaded ADMET model from {path}")
        except Exception as e:
            logger.warning(f"Failed to load ADMET model: {e}")
    
    def predict(self, smiles: str) -> ADMETResult:
        """
        Predict ADMET properties for a molecule
        
        Args:
            smiles: SMILES string
        
        Returns:
            ADMETResult with all predictions
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return self._empty_result(smiles)
        
        if self.model_loaded:
            # Use GNN predictions
            preds = self.model.predict_smiles(smiles)
            if preds is None:
                return self._rule_based_predict(smiles, mol)
            return self._preds_to_result(smiles, preds, 'high')
        else:
            # Use rule-based predictions
            return self._rule_based_predict(smiles, mol)
    
    def _rule_based_predict(self, smiles: str, mol) -> ADMETResult:
        """Rule-based ADMET prediction using molecular descriptors"""
        try:
            # Computed descriptors
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            rotatable = Descriptors.NumRotatableBonds(mol)
            
            # Absorption: Based on Lipinski and permeability
            absorption = 1.0
            if mw > 500: absorption -= 0.3
            if logp > 5 or logp < -2: absorption -= 0.3
            if tpsa > 140: absorption -= 0.3
            if hbd > 5: absorption -= 0.2
            absorption = max(0.0, absorption)
            
            # Distribution: Plasma protein binding and Vd
            distribution = 0.7  # Default moderate
            if logp > 3: distribution += 0.2  # High lipophilicity
            if tpsa < 80: distribution += 0.1
            distribution = min(1.0, distribution)
            
            # Metabolism: CYP liability
            metabolism = 0.6  # Default moderate stability
            if mw > 400: metabolism -= 0.1
            if Chem.MolFromSmarts('[NH2]').GetSubstructMatches(mol): metabolism -= 0.2  # Primary amine
            metabolism = max(0.0, metabolism)
            
            # Excretion: Renal and biliary
            excretion = 0.6
            if mw < 400: excretion += 0.2  # Smaller = easier renal
            excretion = min(1.0, excretion)
            
            # Toxicity risk (structural alerts)
            toxicity_risk = 0.2  # Base risk
            toxic_smarts = [
                '[N+](=O)[O-]',  # Nitro
                'c1ccc2c(c1)ccc1ccccc12',  # PAH
                '[Br,I]c1ccccc1',  # Halogenated aromatic
                'C=O[OH]',  # Aldehyde
            ]
            for smarts in toxic_smarts:
                pattern = Chem.MolFromSmarts(smarts)
                if pattern and mol.HasSubstructMatch(pattern):
                    toxicity_risk += 0.2
            toxicity_risk = min(1.0, toxicity_risk)
            
            # hERG risk (cardiac)
            herg_risk = 0.2
            if logp > 3.5: herg_risk += 0.2
            if tpsa < 75: herg_risk += 0.2
            herg_risk = min(1.0, herg_risk)
            
            # Hepatotoxicity
            hepatotoxicity = 0.2
            if logp > 4: hepatotoxicity += 0.2
            if mw > 500: hepatotoxicity += 0.1
            hepatotoxicity = min(1.0, hepatotoxicity)
            
            # BBB permeability
            bbb = 0.3
            if tpsa < 90 and logp > 0 and logp < 4 and mw < 450:
                bbb = 0.7
            
            # Solubility
            solubility = 0.6
            if logp > 4: solubility -= 0.3
            if logp < 0: solubility += 0.2
            solubility = max(0.0, min(1.0, solubility))
            
            # Overall score: Higher = better drug candidate
            # Invert toxicity scores (lower toxicity = higher score)
            overall = (
                0.20 * absorption +
                0.10 * distribution +
                0.15 * metabolism +
                0.10 * excretion +
                0.20 * (1 - toxicity_risk) +  # Invert
                0.10 * (1 - herg_risk) +       # Invert
                0.10 * (1 - hepatotoxicity) +  # Invert
                0.05 * solubility
            )
            
            return ADMETResult(
                smiles=smiles,
                absorption=absorption,
                distribution=distribution,
                metabolism=metabolism,
                excretion=excretion,
                toxicity_risk=toxicity_risk,
                herg_risk=herg_risk,
                hepatotoxicity=hepatotoxicity,
                bbb_permeability=bbb,
                solubility=solubility,
                confidence='medium',
                overall_score=overall
            )
            
        except Exception as e:
            logger.warning(f"Rule-based ADMET failed for {smiles}: {e}")
            return self._empty_result(smiles)
    
    def _preds_to_result(self, smiles: str, preds: Dict[str, float], conf: str) -> ADMETResult:
        """Convert model predictions to ADMETResult"""
        overall = (
            0.20 * preds.get('absorption', 0.5) +
            0.10 * preds.get('distribution', 0.5) +
            0.15 * preds.get('metabolism', 0.5) +
            0.10 * preds.get('excretion', 0.5) +
            0.20 * (1 - preds.get('toxicity', 0.5)) +
            0.10 * (1 - preds.get('herg', 0.5)) +
            0.10 * (1 - preds.get('hepatotoxicity', 0.5)) +
            0.05 * preds.get('solubility', 0.5)
        )
        
        return ADMETResult(
            smiles=smiles,
            absorption=preds.get('absorption', 0.5),
            distribution=preds.get('distribution', 0.5),
            metabolism=preds.get('metabolism', 0.5),
            excretion=preds.get('excretion', 0.5),
            toxicity_risk=preds.get('toxicity', 0.5),
            herg_risk=preds.get('herg', 0.5),
            hepatotoxicity=preds.get('hepatotoxicity', 0.5),
            bbb_permeability=preds.get('bbb', 0.5),
            solubility=preds.get('solubility', 0.5),
            confidence=conf,
            overall_score=overall
        )
    
    def _empty_result(self, smiles: str) -> ADMETResult:
        """Return empty result for invalid molecules"""
        return ADMETResult(
            smiles=smiles,
            absorption=0.0,
            distribution=0.0,
            metabolism=0.0,
            excretion=0.0,
            toxicity_risk=1.0,
            herg_risk=1.0,
            hepatotoxicity=1.0,
            bbb_permeability=0.0,
            solubility=0.0,
            confidence='low',
            overall_score=0.0
        )
    
    def get_admet_reward(self, smiles: str) -> float:
        """
        Get ADMET reward for RL integration
        
        Returns:
            float: 0-1 score where higher = better ADMET profile
        """
        result = self.predict(smiles)
        return result.overall_score


# Quick test
if __name__ == '__main__':
    agent = ADMETAgent()
    
    test_molecules = {
        'Aspirin': 'CC(=O)Oc1ccccc1C(=O)O',
        'Ibuprofen': 'CC(C)Cc1ccc(cc1)C(C)C(=O)O',
        'Diphenhydramine': 'CN(C)CCOC(c1ccccc1)c1ccccc1',
        'Benzene': 'c1ccccc1',
    }
    
    print("=" * 60)
    print("ADMET PREDICTION TEST")
    print("=" * 60)
    
    for name, smiles in test_molecules.items():
        result = agent.predict(smiles)
        print(f"\n{name}: {smiles}")
        print(f"  Absorption: {result.absorption:.3f}")
        print(f"  Distribution: {result.distribution:.3f}")
        print(f"  Metabolism: {result.metabolism:.3f}")
        print(f"  Toxicity Risk: {result.toxicity_risk:.3f}")
        print(f"  hERG Risk: {result.herg_risk:.3f}")
        print(f"  BBB: {result.bbb_permeability:.3f}")
        print(f"  Solubility: {result.solubility:.3f}")
        print(f"  OVERALL: {result.overall_score:.3f} ({result.confidence})")
    
    print("\n" + "=" * 60)
    print("ADMET Foundation Ready!")
    print("To train on real data: python train_admet.py --dataset toxcast")
    print("=" * 60)
