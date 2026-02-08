"""
Molecule preprocessing and representation conversion for ML models

Converts SMILES to various representations:
- Graph representations for GNNs
- Fingerprints for classical ML
- 3D conformers for docking
- SMILES tokenization for transformers
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MoleculeRepresentation:
    """Container for different molecule representations"""
    smiles: str
    fingerprint: Optional[np.ndarray] = None
    graph: Optional[Dict] = None
    conformer_3d: Optional[np.ndarray] = None
    tokens: Optional[List[str]] = None
    properties: Optional[Dict] = None


class MoleculePreprocessor(ABC):
    """Abstract base class for molecule preprocessing"""

    @abstractmethod
    def process(self, smiles: str) -> MoleculeRepresentation:
        """Process a SMILES string into molecule representation"""
        pass


class RDKitPreprocessor(MoleculePreprocessor):
    """RDKit-based molecule preprocessing"""

    def __init__(self):
        try:
            from rdkit import Chem, DataStructs
            from rdkit.Chem import rdMolDescriptors, Descriptors, AllChem
            self.Chem = Chem
            self.DataStructs = DataStructs
            self.rdMolDescriptors = rdMolDescriptors
            self.Descriptors = Descriptors
            self.AllChem = AllChem
            self.available = True
        except ImportError:
            logger.error("RDKit not available. Install with: pip install rdkit")
            self.available = False

    def process(self, smiles: str) -> MoleculeRepresentation:
        """Process SMILES using RDKit"""
        if not self.available:
            return MoleculeRepresentation(smiles=smiles)
        
        mol = self.Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Invalid SMILES: {smiles}")
            return MoleculeRepresentation(smiles=smiles)
        
        representation = MoleculeRepresentation(smiles=smiles)
        
        # Generate fingerprint
        try:
            representation.fingerprint = self._get_fingerprint(mol)
        except Exception as e:
            logger.warning(f"Fingerprint generation failed for {smiles}: {e}")
        
        # Generate graph representation
        try:
            representation.graph = self._get_graph(mol)
        except Exception as e:
            logger.warning(f"Graph generation failed for {smiles}: {e}")
        
        # Generate 3D conformer
        try:
            representation.conformer_3d = self._get_3d_conformer(mol)
        except Exception as e:
            logger.warning(f"3D conformer generation failed for {smiles}: {e}")
        
        # Calculate properties
        try:
            representation.properties = self._get_properties(mol)
        except Exception as e:
            logger.warning(f"Property calculation failed for {smiles}: {e}")
        
        return representation

    def _get_fingerprint(self, mol, fp_type: str = 'morgan', radius: int = 2,
                        n_bits: int = 2048) -> np.ndarray:
        """Generate molecular fingerprint"""
        if fp_type == 'morgan':
            fp = self.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        elif fp_type == 'rdkit':
            fp = self.Chem.RDKFingerprint(mol, fpSize=n_bits)
        else:
            raise ValueError(f"Unsupported fingerprint type: {fp_type}")
        
        array = np.zeros((n_bits,), dtype=np.int8)
        self.DataStructs.ConvertToNumpyArray(fp, array)
        return array

    def _get_graph(self, mol) -> Dict:
        """Convert molecule to graph representation"""
        atom_features = []
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                int(atom.GetHybridization()),
                int(atom.GetIsAromatic()),
                atom.GetTotalNumHs(),
                int(atom.IsInRing()),
            ]
            atom_features.append(features)
        
        edges = []
        edge_features = []
        for bond in mol.GetBonds():
            start_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            
            edges.extend([(start_idx, end_idx), (end_idx, start_idx)])
            
            bond_feat = [
                int(bond.GetBondType()),
                int(bond.GetIsConjugated()),
                int(bond.IsInRing())
            ]
            edge_features.extend([bond_feat, bond_feat])
        
        return {
            'atom_features': np.array(atom_features, dtype=np.float32),
            'edge_indices': np.array(edges, dtype=np.int64).T if edges else np.empty((2, 0), dtype=np.int64),
            'edge_features': np.array(edge_features, dtype=np.float32) if edge_features else np.empty((0, 3), dtype=np.float32),
            'num_atoms': mol.GetNumAtoms(),
            'num_bonds': mol.GetNumBonds()
        }

    def _get_3d_conformer(self, mol) -> Optional[np.ndarray]:
        """Generate 3D conformer"""
        mol_copy = self.Chem.Mol(mol)
        mol_copy = self.Chem.AddHs(mol_copy)
        
        result = self.AllChem.EmbedMolecule(mol_copy, randomSeed=42)
        if result != 0:
            logger.warning("3D embedding failed")
            return None
        
        self.AllChem.MMFFOptimizeMolecule(mol_copy)
        
        conformer = mol_copy.GetConformer()
        coords = []
        for i in range(mol_copy.GetNumAtoms()):
            pos = conformer.GetAtomPosition(i)
            coords.append([pos.x, pos.y, pos.z])
        
        return np.array(coords, dtype=np.float32)

    def _get_properties(self, mol) -> Dict:
        """Calculate molecular properties"""
        return {
            'molecular_weight': self.Descriptors.MolWt(mol),
            'logp': self.Descriptors.MolLogP(mol),
            'hbd': self.Descriptors.NumHDonors(mol),
            'hba': self.Descriptors.NumHAcceptors(mol),
            'tpsa': self.Descriptors.TPSA(mol),
            'rotatable_bonds': self.Descriptors.NumRotatableBonds(mol),
            'aromatic_rings': self.Descriptors.NumAromaticRings(mol),
            'heavy_atoms': mol.GetNumHeavyAtoms(),
            'formal_charge': self.Chem.GetFormalCharge(mol)
        }


class SMILESTokenizer:
    """Tokenize SMILES strings for transformer models"""

    def __init__(self):
        self.atoms = [
            'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
            'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn',
            'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au',
            'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb'
        ]
        
        self.special_tokens = [
            '(', ')', '[', ']', '=', '#', '@', '*', '%', '0', '1', '2', '3', '4',
            '5', '6', '7', '8', '9', '+', '-', '\\', '/', ':',
            'c', 'n', 'o', 's'
        ]
        
        self.vocab = ['<PAD>', '<START>', '<END>', '<UNK>'] + self.atoms + self.special_tokens
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocab)}

    def tokenize(self, smiles: str) -> List[str]:
        """Tokenize SMILES string"""
        tokens = []
        i = 0
        while i < len(smiles):
            if smiles[i] == '[':
                j = smiles.find(']', i)
                if j != -1:
                    tokens.append(smiles[i:j + 1])
                    i = j + 1
                else:
                    tokens.append(smiles[i])
                    i += 1
            elif smiles[i:i + 2] in ['Cl', 'Br', 'Si']:
                tokens.append(smiles[i:i + 2])
                i += 2
            else:
                tokens.append(smiles[i])
                i += 1
        
        return tokens

    def encode(self, smiles: str, max_length: int = 128) -> List[int]:
        """Encode SMILES to token IDs"""
        tokens = self.tokenize(smiles)
        token_ids = [self.token_to_id.get('<START>')]
        
        for token in tokens:
            token_id = self.token_to_id.get(token, self.token_to_id.get('<UNK>'))
            token_ids.append(token_id)
        
        token_ids.append(self.token_to_id.get('<END>'))
        
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
            token_ids[-1] = self.token_to_id.get('<END>')
        else:
            token_ids.extend([self.token_to_id.get('<PAD>')] * (max_length - len(token_ids)))
        
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to SMILES"""
        tokens = []
        for token_id in token_ids:
            token = self.id_to_token.get(token_id, '<UNK>')
            if token in ['<PAD>', '<START>', '<END>']:
                continue
            elif token == '<UNK>':
                break
            else:
                tokens.append(token)
        
        return ''.join(tokens)


class BatchPreprocessor:
    """Batch processing for efficient preprocessing"""

    def __init__(self, preprocessor: MoleculePreprocessor = None):
        self.preprocessor = preprocessor or RDKitPreprocessor()
        self.tokenizer = SMILESTokenizer()

    def process_batch(self, smiles_list: List[str],
                     include_fingerprints: bool = True,
                     include_graphs: bool = True,
                     include_3d: bool = False,
                     include_tokens: bool = True) -> Dict[str, List]:
        """Process batch of SMILES strings"""
        results = {
            'smiles': [],
            'fingerprints': [],
            'graphs': [],
            'conformers_3d': [],
            'tokens': [],
            'properties': []
        }
        
        for smiles in smiles_list:
            try:
                representation = self.preprocessor.process(smiles)
                
                results['smiles'].append(smiles)
                
                if include_fingerprints:
                    results['fingerprints'].append(representation.fingerprint)
                
                if include_graphs:
                    results['graphs'].append(representation.graph)
                
                if include_3d:
                    results['conformers_3d'].append(representation.conformer_3d)
                
                if include_tokens:
                    tokens = self.tokenizer.encode(smiles)
                    results['tokens'].append(tokens)
                
                results['properties'].append(representation.properties)
                
            except Exception as e:
                logger.error(f"Failed to process {smiles}: {e}")
                results['smiles'].append(smiles)
                results['fingerprints'].append(None)
                results['graphs'].append(None)
                results['conformers_3d'].append(None)
                results['tokens'].append(None)
                results['properties'].append(None)
        
        logger.info(f"Successfully processed {len(results['smiles'])}/{len(smiles_list)} molecules")
        return results

    def create_feature_matrix(self, fingerprints: List[np.ndarray]) -> np.ndarray:
        """Create feature matrix from fingerprints"""
        valid_fps = [fp for fp in fingerprints if fp is not None]
        if not valid_fps:
            return np.array([])
        return np.vstack(valid_fps)

    def create_graph_batch(self, graphs: List[Dict]) -> Dict:
        """Create batched graph data for GNNs"""
        if not graphs or all(g is None for g in graphs):
            return {}
        
        valid_graphs = [g for g in graphs if g is not None]
        
        batch_atom_features = []
        batch_edge_indices = []
        batch_edge_features = []
        batch_indices = []
        
        atom_offset = 0
        for i, graph in enumerate(valid_graphs):
            batch_atom_features.append(graph['atom_features'])
            
            edge_indices = graph['edge_indices'] + atom_offset
            batch_edge_indices.append(edge_indices)
            batch_edge_features.append(graph['edge_features'])
            batch_indices.extend([i] * graph['num_atoms'])
            
            atom_offset += graph['num_atoms']
        
        return {
            'atom_features': np.vstack(batch_atom_features) if batch_atom_features else np.array([]),
            'edge_indices': np.hstack(batch_edge_indices) if batch_edge_indices else np.array([]),
            'edge_features': np.vstack(batch_edge_features) if batch_edge_features else np.array([]),
            'batch_indices': np.array(batch_indices),
            'num_graphs': len(valid_graphs)
        }


if __name__ == "__main__":
    test_smiles = [
        'CCO',
        'CC(=O)O',
        'c1ccccc1',
        'CCN(CC)CC',
    ]
    
    preprocessor = BatchPreprocessor()
    results = preprocessor.process_batch(test_smiles)
    
    print(f"Processed {len(results['smiles'])} molecules")
    
    if results['fingerprints']:
        features = preprocessor.create_feature_matrix(results['fingerprints'])
        print(f"Feature matrix shape: {features.shape}")
    
    if results['graphs']:
        graph_batch = preprocessor.create_graph_batch(results['graphs'])
        print(f"Graph batch with {graph_batch.get('num_graphs', 0)} molecules")
