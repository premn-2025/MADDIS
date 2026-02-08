"""
Property Prediction Agent for Multi-Agent Drug Discovery

This agent uses Graph Neural Networks (GNN) to predict molecular properties
critical for drug development:
    - Toxicity (acute toxicity, mutagenicity)
    - Solubility (aqueous solubility)
    - Permeability (Caco-2, MDCK)
    - Blood-Brain Barrier (BBB) permeability
    - Drug-likeness (QED, Lipinski's Rule of 5)

Architecture: AttentiveFP (Attentive Fingerprints for molecular property prediction)
Training: Multi-task learning on ToxCast, AqSolDB, and other ADMET datasets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors, Descriptors, Crippen, QED
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
import logging
from typing import Dict, List, Tuple, Optional, Any
import pickle
import os
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Try to import torch-geometric, fall back to basic implementation if not available
try:
    from torch_geometric.nn import GATConv, global_add_pool, global_mean_pool, global_max_pool
    from torch_geometric.data import Data, Batch
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    print("torch-geometric not available. Using basic implementation.")
    HAS_TORCH_GEOMETRIC = False

    # Mock Data class for compatibility
    class Data:
        def __init__(self, x=None, edge_index=None, edge_attr=None, batch=None):
            self.x = x
            self.edge_index = edge_index
            self.edge_attr = edge_attr
            self.batch = batch

        def to(self, device):
            if self.x is not None:
                self.x = self.x.to(device)
            if self.edge_index is not None:
                self.edge_index = self.edge_index.to(device)
            if self.edge_attr is not None:
                self.edge_attr = self.edge_attr.to(device)
            if self.batch is not None:
                self.batch = self.batch.to(device)
            return self

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PropertyPrediction:
    """Container for molecular property predictions"""
    smiles: str
    
    # Core ADMET properties
    toxicity_score: float  # 0-1 (1 = toxic)
    solubility: float  # LogS (higher = more soluble)
    permeability: float  # LogPerm (higher = more permeable)
    bbb_permeability: float  # 0-1 (1 = crosses BBB)
    
    # Drug-likeness scores
    qed_score: float  # 0-1 (1 = drug-like)
    lipinski_violations: int  # 0-4 violations
    synthetic_accessibility: float  # 1-10 (1 = easy to synthesize)
    
    # Computed properties
    molecular_weight: float
    logp: float
    hbd: int  # Hydrogen bond donors
    hba: int  # Hydrogen bond acceptors
    rotatable_bonds: int
    tpsa: float  # Topological polar surface area
    
    # Overall scores
    overall_drug_score: float  # Combined score (0-1)
    is_drug_like: bool
    warnings: List[str]


class MolecularGraph:
    """Convert SMILES to graph representation for GNN"""
    
    def __init__(self):
        # Atom features (one-hot encoded)
        self.atom_symbols = [
            'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
            'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag',
            'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni',
            'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'
        ]
        self.degrees = [0, 1, 2, 3, 4, 5]
        self.formal_charges = [-1, -2, 1, 2, 0]
        self.hybridizations = [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2
        ]
    
    def smiles_to_graph(self, smiles: str) -> Optional[Data]:
        """Convert SMILES string to PyTorch Geometric Data object"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Get atom features
        atom_features = []
        for atom in mol.GetAtoms():
            features = []
            
            # Atom symbol (one-hot) - 44 features
            atom_symbol = atom.GetSymbol()
            features.extend([atom_symbol == s for s in self.atom_symbols])
            
            # Degree (one-hot) - 6 features
            degree = atom.GetDegree()
            features.extend([degree == d for d in self.degrees])
            
            # Formal charge (one-hot) - 5 features
            charge = atom.GetFormalCharge()
            features.extend([charge == c for c in self.formal_charges])
            
            # Hybridization (one-hot) - 5 features
            hybridization = atom.GetHybridization()
            features.extend([hybridization == h for h in self.hybridizations])
            
            # Additional features - 4 features
            features.append(atom.GetIsAromatic())
            features.append(atom.IsInRing())
            features.append(atom.GetTotalValence())
            features.append(atom.GetDegree())
            
            atom_features.append(features)
        
        # Get edge information
        edge_indices = []
        edge_features = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            # Add both directions
            edge_indices.extend([[i, j], [j, i]])
            
            # Bond features (repeated for both directions)
            bond_type = bond.GetBondType()
            features = [
                bond_type == Chem.rdchem.BondType.SINGLE,
                bond_type == Chem.rdchem.BondType.DOUBLE,
                bond_type == Chem.rdchem.BondType.TRIPLE,
                bond_type == Chem.rdchem.BondType.AROMATIC,
                bond.GetIsConjugated(),
                bond.IsInRing()
            ]
            edge_features.extend([features, features])
        
        # Convert to tensors
        x = torch.tensor(atom_features, dtype=torch.float)
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous() if edge_indices else torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.tensor(edge_features, dtype=torch.float) if edge_features else torch.empty((0, 6), dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class AttentiveFP(nn.Module):
    """
    AttentiveFP: Attentive Fingerprints for molecular property prediction
    
    Fallback implementation without torch-geometric
    """
    
    def __init__(self,
                 node_feat_size: int = 64,
                 edge_feat_size: int = 6,
                 hidden_size: int = 200,
                 num_layers: int = 4,
                 num_tasks: int = 5,
                 dropout: float = 0.1):
        super(AttentiveFP, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.has_torch_geometric = HAS_TORCH_GEOMETRIC
        
        # Input projection
        self.node_embedding = nn.Linear(node_feat_size, hidden_size)
        self.edge_embedding = nn.Linear(edge_feat_size, hidden_size)
        
        if HAS_TORCH_GEOMETRIC:
            # Graph Attention Layers
            self.attention_layers = nn.ModuleList([
                GATConv(hidden_size, hidden_size // 8, heads=8, dropout=dropout, edge_dim=hidden_size)
                for _ in range(num_layers)
            ])
        else:
            # Fallback: Simple MLPs without graph structure
            self.attention_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
                for _ in range(num_layers)
            ])
        
        # Readout layers
        self.readout_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size)
            for _ in range(num_layers)
        ])
        
        # Final prediction heads (multi-task)
        self.task_heads = nn.ModuleDict({
            'toxicity': nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 1),
                nn.Sigmoid()
            ),
            'solubility': nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 1)
            ),
            'permeability': nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 1)
            ),
            'bbb_permeability': nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 1),
                nn.Sigmoid()
            ),
            'qed': nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 1),
                nn.Sigmoid()
            )
        })
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, data):
        if HAS_TORCH_GEOMETRIC:
            return self._forward_with_gnn(data)
        else:
            return self._forward_fallback(data)
    
    def _forward_with_gnn(self, data):
        """Forward pass with proper graph neural network"""
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Initial embeddings
        h = self.node_embedding(x)
        edge_h = self.edge_embedding(edge_attr) if edge_attr.size(0) > 0 else None
        
        # Store representations from each layer
        readouts = []
        
        for i, (attention_layer, readout_layer) in enumerate(zip(self.attention_layers, self.readout_layers)):
            # Graph attention
            h = attention_layer(h, edge_index, edge_attr=edge_h)
            h = self.dropout(h)
            
            # Global readout (sum pooling)
            readout = readout_layer(global_add_pool(h, batch))
            readouts.append(readout)
        
        # Combine readouts from all layers
        graph_repr = sum(readouts)
        
        # Multi-task predictions
        predictions = {}
        for task_name, head in self.task_heads.items():
            predictions[task_name] = head(graph_repr)
        
        return predictions
    
    def _forward_fallback(self, data):
        """Fallback forward pass without graph structure"""
        x = data.x
        
        # Initial embedding
        h = self.node_embedding(x)
        
        # Simple layer processing without graph structure
        readouts = []
        for i, (attention_layer, readout_layer) in enumerate(zip(self.attention_layers, self.readout_layers)):
            h = attention_layer(h)
            
            # Simple mean pooling over nodes
            pooled = torch.mean(h, dim=0, keepdim=True)
            readout = readout_layer(pooled)
            readouts.append(readout)
        
        # Combine readouts
        graph_repr = sum(readouts)
        
        # Multi-task predictions
        predictions = {}
        for task_name, head in self.task_heads.items():
            predictions[task_name] = head(graph_repr)
        
        return predictions


class PropertyPredictionAgent:
    """
    Advanced property prediction agent using Graph Neural Networks
    
    Predicts critical ADMET properties for drug development:
    - Toxicity, Solubility, Permeability, BBB crossing, Drug-likeness
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Property Predictor initialized on {self.device}")
        
        # Initialize molecular graph converter
        self.graph_converter = MolecularGraph()
        
        # Calculate actual feature size
        actual_node_feat_size = (
            len(self.graph_converter.atom_symbols) + 
            len(self.graph_converter.degrees) + 
            len(self.graph_converter.formal_charges) + 
            len(self.graph_converter.hybridizations) + 4
        )
        
        # Initialize model
        self.model = AttentiveFP(
            node_feat_size=actual_node_feat_size,
            edge_feat_size=6,
            hidden_size=200,
            num_layers=4,
            num_tasks=5,
            dropout=0.1
        ).to(self.device)
        
        # Load pre-trained weights if available
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            logger.info(f"Loaded pre-trained model from {model_path}")
        else:
            logger.warning("No pre-trained model found. Using randomly initialized weights.")
        
        # Initialize drug-likeness filters
        self._setup_drug_filters()
        
        logger.info("Property Prediction Agent ready for inference")
    
    def _setup_drug_filters(self):
        """Setup molecular filters for drug-likeness assessment"""
        # PAINS (Pan-assay interference compounds) filter
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
        self.pains_catalog = FilterCatalog(params)
        
        logger.info("Drug-likeness filters initialized")
    
    def predict_properties(self, smiles: str) -> PropertyPrediction:
        """
        Predict comprehensive molecular properties for a SMILES string
        
        Args:
            smiles: SMILES string of the molecule
            
        Returns:
            PropertyPrediction object with all predicted properties
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return self._create_invalid_prediction(smiles, ["Invalid SMILES"])
        
        try:
            # Convert to graph
            graph_data = self.graph_converter.smiles_to_graph(smiles)
            if graph_data is None:
                return self._create_invalid_prediction(smiles, ["Cannot convert to graph"])
            
            # GNN predictions
            gnn_predictions = self._predict_with_gnn(graph_data)
            
            # RDKit computed properties
            computed_props = self._compute_rdkit_properties(mol)
            
            # Drug-likeness assessment
            drug_assessment = self._assess_drug_likeness(mol)
            
            # Combine all predictions
            prediction = PropertyPrediction(
                smiles=smiles,
                toxicity_score=gnn_predictions.get('toxicity', 0.5),
                solubility=gnn_predictions.get('solubility', 0.0),
                permeability=gnn_predictions.get('permeability', 0.0),
                bbb_permeability=gnn_predictions.get('bbb_permeability', 0.5),
                qed_score=gnn_predictions.get('qed', computed_props['qed']),
                molecular_weight=computed_props['mw'],
                logp=computed_props['logp'],
                hbd=computed_props['hbd'],
                hba=computed_props['hba'],
                rotatable_bonds=computed_props['rotatable_bonds'],
                tpsa=computed_props['tpsa'],
                lipinski_violations=drug_assessment['lipinski_violations'],
                synthetic_accessibility=drug_assessment['sa_score'],
                overall_drug_score=self._calculate_overall_score(gnn_predictions, computed_props, drug_assessment),
                is_drug_like=drug_assessment['is_drug_like'],
                warnings=drug_assessment['warnings']
            )
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting properties for {smiles}: {e}")
            return self._create_invalid_prediction(smiles, [str(e)])
    
    def _predict_with_gnn(self, graph_data: Data) -> Dict[str, float]:
        """Run GNN inference on graph data"""
        self.model.eval()
        
        with torch.no_grad():
            # Add batch dimension
            graph_data = graph_data.to(self.device)
            graph_data.batch = torch.zeros(graph_data.x.size(0), dtype=torch.long, device=self.device)
            
            # Forward pass
            predictions = self.model(graph_data)
            
            # Extract scalar values
            results = {}
            for task, pred_tensor in predictions.items():
                results[task] = pred_tensor.cpu().item()
            
            return results
    
    def _compute_rdkit_properties(self, mol) -> Dict[str, float]:
        """Compute molecular descriptors using RDKit"""
        try:
            return {
                'mw': Descriptors.MolWt(mol),
                'logp': Crippen.MolLogP(mol),
                'hbd': rdMolDescriptors.CalcNumHBD(mol),
                'hba': rdMolDescriptors.CalcNumHBA(mol),
                'rotatable_bonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
                'tpsa': rdMolDescriptors.CalcTPSA(mol),
                'qed': QED.qed(mol)
            }
        except:
            return {
                'mw': 0.0, 'logp': 0.0, 'hbd': 0, 'hba': 0,
                'rotatable_bonds': 0, 'tpsa': 0.0, 'qed': 0.0
            }
    
    def _assess_drug_likeness(self, mol) -> Dict[str, Any]:
        """Assess drug-likeness using various rules and filters"""
        warnings = []
        
        # Lipinski's Rule of 5
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        
        lipinski_violations = 0
        if mw > 500:
            lipinski_violations += 1
            warnings.append("MW > 500 Da")
        if logp > 5:
            lipinski_violations += 1
            warnings.append("LogP > 5")
        if hbd > 5:
            lipinski_violations += 1
            warnings.append("H-bond donors > 5")
        if hba > 10:
            lipinski_violations += 1
            warnings.append("H-bond acceptors > 10")
        
        # PAINS filter
        if self.pains_catalog.HasMatch(mol):
            warnings.append("PAINS compound (promiscuous)")
        
        # Synthetic Accessibility Score (approximation)
        sa_score = min(10, max(1, 1 + (mw - 150) / 100 + abs(logp) + lipinski_violations))
        
        # Overall drug-likeness
        is_drug_like = (
            lipinski_violations <= 1 and
            len(warnings) == 0 and
            150 <= mw <= 600 and
            -2 <= logp <= 6
        )
        
        return {
            'lipinski_violations': lipinski_violations,
            'sa_score': sa_score,
            'is_drug_like': is_drug_like,
            'warnings': warnings
        }
    
    def _calculate_overall_score(self, gnn_pred: Dict, computed: Dict, assessment: Dict) -> float:
        """Calculate overall drug-likeness score (0-1)"""
        factors = []
        
        # Toxicity (lower is better)
        factors.append(1.0 - gnn_pred.get('toxicity', 0.5))
        
        # QED score (higher is better)
        factors.append(max(computed.get('qed', 0.0), gnn_pred.get('qed', 0.0)))
        
        # Lipinski compliance (fewer violations is better)
        lipinski_score = max(0, 1.0 - assessment['lipinski_violations'] / 4.0)
        factors.append(lipinski_score)
        
        # Synthetic accessibility (lower is better, normalize to 0-1)
        sa_score = max(0, 1.0 - (assessment['sa_score'] - 1) / 9.0)
        factors.append(sa_score)
        
        # PAINS penalty
        pains_penalty = len([w for w in assessment['warnings'] if 'PAINS' in w]) * 0.3
        
        overall = np.mean(factors) - pains_penalty
        return max(0.0, min(1.0, overall))
    
    def _create_invalid_prediction(self, smiles: str, warnings: List[str]) -> PropertyPrediction:
        """Create prediction object for invalid molecules"""
        return PropertyPrediction(
            smiles=smiles,
            toxicity_score=1.0,
            solubility=-10.0,
            permeability=-10.0,
            bbb_permeability=0.0,
            qed_score=0.0,
            lipinski_violations=4,
            synthetic_accessibility=10.0,
            molecular_weight=0.0,
            logp=0.0,
            hbd=0,
            hba=0,
            rotatable_bonds=0,
            tpsa=0.0,
            overall_drug_score=0.0,
            is_drug_like=False,
            warnings=warnings
        )
    
    def filter_molecules(
        self,
        molecules: List[str],
        min_drug_score: float = 0.5,
        max_toxicity: float = 0.3,
        max_lipinski_violations: int = 1
    ) -> List[Tuple[str, PropertyPrediction]]:
        """
        Filter molecules based on drug-likeness criteria
        
        Args:
            molecules: List of SMILES strings
            min_drug_score: Minimum overall drug score (0-1)
            max_toxicity: Maximum toxicity score (0-1)
            max_lipinski_violations: Maximum Lipinski violations allowed
            
        Returns:
            List of (SMILES, PropertyPrediction) tuples for passing molecules
        """
        logger.info(f"Filtering {len(molecules)} molecules for drug-likeness...")
        
        passing_molecules = []
        for smiles in molecules:
            pred = self.predict_properties(smiles)
            
            # Apply filters
            if (pred.overall_drug_score >= min_drug_score and
                pred.toxicity_score <= max_toxicity and
                pred.lipinski_violations <= max_lipinski_violations and
                pred.is_drug_like):
                passing_molecules.append((smiles, pred))
        
        logger.info(f"{len(passing_molecules)}/{len(molecules)} molecules passed filtering "
                   f"({100*len(passing_molecules)/len(molecules):.1f}%)")
        
        return passing_molecules
    
    def save_model(self, path: str):
        """Save trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'node_feat_size': 64,
                'edge_feat_size': 6,
                'hidden_size': 200,
                'num_layers': 4,
                'num_tasks': 5,
                'dropout': 0.1
            }
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load pre-trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {path}")


def test_property_predictor():
    """Test the property prediction agent"""
    logger.info("Testing Property Prediction Agent...")
    
    # Initialize agent
    agent = PropertyPredictionAgent()
    
    # Test molecules
    test_molecules = [
        "CCO",  # Ethanol (simple, safe)
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC(C)Cc1ccc(C(C)C(=O)O)cc1",  # Ibuprofen
    ]
    
    print("\n" + "="*60)
    print("Property Prediction Results")
    print("="*60)
    
    for smiles in test_molecules:
        pred = agent.predict_properties(smiles)
        print(f"\nMolecule: {smiles[:40]}...")
        print(f"  Toxicity: {pred.toxicity_score:.3f}")
        print(f"  Solubility: {pred.solubility:.3f}")
        print(f"  QED Score: {pred.qed_score:.3f}")
        print(f"  Overall Drug Score: {pred.overall_drug_score:.3f}")
        print(f"  Drug-like: {pred.is_drug_like}")
        if pred.warnings:
            print(f"  Warnings: {', '.join(pred.warnings)}")
    
    print("\n" + "="*60)
    print("Test complete!")


if __name__ == "__main__":
    test_property_predictor()
