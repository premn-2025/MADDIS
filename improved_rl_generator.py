#!/usr/bin/env python3
"""
IMPROVED RL MOLECULAR GENERATOR WITH VALID SMILES

This version uses a grammar-guided approach and pre-trained starting points
to ensure valid molecular generation.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import asyncio
import random

# RDKit for molecular operations
from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, Descriptors, QED

# Synthetic Accessibility Score
try:
    import sys
    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    import sascorer
    SA_SCORE_AVAILABLE = True
except ImportError:
    SA_SCORE_AVAILABLE = False
    sascorer = None

# Import our docking system
try:
    from real_docking_agent import RealMolecularDockingAgent
    DOCKING_AVAILABLE = True
except ImportError:
    DOCKING_AVAILABLE = False

# Import ADMET predictor
try:
    from admet_predictor import ADMETAgent
    ADMET_AVAILABLE = True
except ImportError:
    ADMET_AVAILABLE = False
    ADMETAgent = None

# Import Neural SMILES Generator
logger = logging.getLogger(__name__)

try:
    from neural_smiles_generator import NeuralSMILESGenerator
    NEURAL_GEN_AVAILABLE = True
except ImportError:
    NEURAL_GEN_AVAILABLE = False
    NeuralSMILESGenerator = None
    logger.warning("Neural SMILES generator not available, using fallback")

# Import Chemical Safety Filter
try:
    from chemical_safety_filter import ChemicalSafetyFilter, check_reactive_groups
    SAFETY_FILTER_AVAILABLE = True
except ImportError:
    SAFETY_FILTER_AVAILABLE = False
    ChemicalSafetyFilter = None
    check_reactive_groups = lambda x: True  # Fallback: accept all
    logger.warning("Chemical safety filter not available")

# Import QSAR Predictor (REAL binding prediction from ChEMBL-trained models)
try:
    from qsar_predictor import QSARPredictor
    QSAR_AVAILABLE = True
except ImportError:
    QSAR_AVAILABLE = False
    QSARPredictor = None
    logger.warning("QSAR predictor not available - run download_chembl_data.py and train_qsar_models.py")



class ValidSMILESGenerator:
    """
    Grammar-guided SMILES generator that produces valid molecules
    """

    def __init__(self):
        # Predefined molecular fragments and scaffolds
        self.core_scaffolds = [
            "c1ccccc1",  # Benzene
            "c1ccc2[nH]c3ccccc3c2c1",  # Carbazole
            "c1cc2ccccc2cc1",  # Naphthalene
            "C1CCC(CC1)",  # Cyclohexane
            "c1cncc2ccccc12",  # Quinoline
            "c1ccc2nc3ccccc3nc2c1",  # Phenanthroline
        ]

        self.functional_groups = [
            "N",  # Amine
            "O",  # Ether/Alcohol
            "C(=O)O",  # Carboxylic acid
            "C(=O)N",  # Amide
            "S(=O)(=O)N",  # Sulfonamide
            "C#N",  # Nitrile
            "C(F)(F)F",  # Trifluoromethyl
            "Cl",  # Chlorine
            "Br",  # Bromine
            "I",  # Iodine
        ]

        self.linkers = [
            "C",  # Methylene - REMOVED empty string which causes dots!
            "CC",  # Ethylene
            "O",  # Ether
            "N",  # Amine
            "C(=O)",  # Carbonyl
        ]

        # Known drug molecules as starting points
        self.known_drugs = [
            "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
            "CC(C)Cc1ccc(C(C)C(=O)O)cc1",  # Ibuprofen
            "CN(C)CCOC(c1ccccc1)c1ccccn1",  # Diphenhydramine
            "COc1cc2[nH]c3c(c2cc1OC)CCN3C",  # Mescaline derivative
            "c1ccc(CCN)cc1",  # Phenethylamine
            "CC(C)(C)c1ccc(OCCCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4)cc1",  # Terfenadine-like
        ]

    def generate_molecule_by_modification(self, base_smiles: str = None, num_modifications: int = 3) -> str:
        """Generate molecule by modifying a known drug"""

        if base_smiles is None:
            base_smiles = random.choice(self.known_drugs)

        mol = Chem.MolFromSmiles(base_smiles)
        if mol is None:
            return random.choice(self.known_drugs)

        # Apply random modifications
        for _ in range(num_modifications):
            mod_type = random.choice(['add_fragment', 'replace_atom', 'add_ring'])

            try:
                if mod_type == 'add_fragment':
                    mol = self._add_functional_group(mol)
                elif mod_type == 'replace_atom':
                    mol = self._replace_random_atom(mol)
                elif mod_type == 'add_ring':
                    mol = self._add_ring_system(mol)

                if mol is None:
                    break
            except Exception:
                break

        if mol is not None:
            return Chem.MolToSmiles(mol)
        else:
            return base_smiles

    def _add_functional_group(self, mol):
        """Add a functional group to a random position"""
        if mol.GetNumAtoms() >= 50:  # Prevent too large molecules
            return mol

        # Find carbon atoms that can be modified
        carbon_atoms = []
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'C' and atom.GetDegree() < 4:
                carbon_atoms.append(atom.GetIdx())

        if not carbon_atoms:
            return mol

        # Add a functional group
        fg = random.choice(self.functional_groups)

        # Simple addition by creating new molecule string
        base_smiles = Chem.MolToSmiles(mol)

        # Try to add functional group (simplified approach)
        if fg == "N":
            new_smiles = base_smiles.replace("C)", "CN)", 1)
        elif fg == "O":
            new_smiles = base_smiles.replace("C)", "CO)", 1)
        elif fg == "Cl":
            new_smiles = base_smiles.replace("C)", "CCl)", 1)
        else:
            new_smiles = base_smiles

        new_mol = Chem.MolFromSmiles(new_smiles)
        return new_mol if new_mol is not None else mol

    def _replace_random_atom(self, mol):
        """Replace a random atom with a similar one"""
        replacements = {
            'C': ['N', 'O', 'S'],
            'N': ['C', 'O'],
            'O': ['N', 'S'],
            'S': ['O', 'N']
        }

        atoms_to_replace = []
        for atom in mol.GetAtoms():
            if atom.GetSymbol() in replacements:
                atoms_to_replace.append((atom.GetIdx(), atom.GetSymbol()))

        if not atoms_to_replace:
            return mol

        # Choose random atom to replace
        atom_idx, current_symbol = random.choice(atoms_to_replace)
        new_symbol = random.choice(replacements[current_symbol])

        # Create new molecule with replacement (simplified)
        smiles = Chem.MolToSmiles(mol)

        # Very basic replacement (not comprehensive)
        if current_symbol == 'C' and new_symbol == 'N':
            new_smiles = smiles.replace('C', 'N', 1)
        else:
            new_smiles = smiles

        new_mol = Chem.MolFromSmiles(new_smiles)
        return new_mol if new_mol is not None else mol

    def _add_ring_system(self, mol):
        """Add a ring system to the molecule"""
        if mol.GetNumAtoms() >= 40:  # Prevent too large molecules
            return mol

        # Add a simple ring (very basic implementation)
        base_smiles = Chem.MolToSmiles(mol)
        ring = random.choice(["c1ccccc1", "C1CCCCC1"])  # Benzene or cyclohexane

        # Simple concatenation with linker - NEVER create dots!
        linker = random.choice(self.linkers)
        new_smiles = f"{base_smiles}{linker}{ring}"

        new_mol = Chem.MolFromSmiles(new_smiles)
        
        # CRITICAL: Reject if result has dots (multiple fragments)
        if new_mol is not None:
            final_smiles = Chem.MolToSmiles(new_mol)
            if '.' not in final_smiles:
                return new_mol
        return mol

    def generate_scaffold_molecule(self) -> str:
        """Generate molecule from scaffold + functional groups"""
        scaffold = random.choice(self.core_scaffolds)

        # Add 1-3 functional groups
        num_groups = random.randint(1, 3)

        mol = Chem.MolFromSmiles(scaffold)
        if mol is None:
            return scaffold

        for _ in range(num_groups):
            mol = self._add_functional_group(mol)
            if mol is None:
                break

        return Chem.MolToSmiles(mol) if mol else scaffold


class ImprovedRLMolecularGenerator:
    """
    Improved RL Molecular Generator with Valid SMILES

    Uses scaffold-based generation and grammar constraints
    """

    def __init__(self, target_protein: str = "COX2", device: str = "auto", use_neural: bool = True):
        self.target_protein = target_protein
        self.device = self._setup_device(device)
        self.use_neural = use_neural and NEURAL_GEN_AVAILABLE

        # Initialize SMILES generator (neural or fallback)
        if self.use_neural:
            self.smiles_generator = NeuralSMILESGenerator(device=device)
            logger.info("✓ Using Neural LSTM SMILES Generator")
        else:
            self.smiles_generator = ValidSMILESGenerator()
            logger.info("Using scaffold-based generator (fallback)")

        # RL parameters - HIGH EXPLORATION to prevent mode collapse
        self.learning_rate = 0.001
        self.epsilon = 0.6  # Start with 60% exploration for neural gen
        self.epsilon_decay = 0.98  # Slower decay
        self.min_epsilon = 0.2  # Keep at least 20% exploration

        # Reward function with real docking
        self.reward_cache = {}

        if DOCKING_AVAILABLE:
            self.docking_agent = RealMolecularDockingAgent()
            self.has_docking = True
            logger.info(" Real docking engine available!")
        else:
            self.docking_agent = None
            self.has_docking = False
            logger.warning(" Using simulated docking rewards")

        # ADMET prediction
        if ADMET_AVAILABLE:
            self.admet_agent = ADMETAgent()
            self.has_admet = True
            logger.info(" ADMET predictor available!")
        else:
            self.admet_agent = None
            self.has_admet = False
            logger.warning(" ADMET prediction not available")

        # QSAR predictor (REAL binding from ChEMBL-trained models)
        if QSAR_AVAILABLE:
            try:
                self.qsar_predictor = QSARPredictor()
                self.has_qsar = self.qsar_predictor.is_available
                if self.has_qsar:
                    logger.info(f" QSAR binding predictor active! Targets: {self.qsar_predictor.available_targets}")
                else:
                    logger.warning(" QSAR models not trained yet - using fallback docking")
            except Exception as e:
                self.qsar_predictor = None
                self.has_qsar = False
                logger.warning(f" QSAR predictor init failed: {e}")
        else:
            self.qsar_predictor = None
            self.has_qsar = False


        # Chemical Safety Filter
        if SAFETY_FILTER_AVAILABLE:
            self.safety_filter = ChemicalSafetyFilter()
            logger.info("Chemical safety filter active")
        else:
            self.safety_filter = None
            logger.warning("Chemical safety filter not available")
        # Training statistics
        self.stats = {
            'generation': 0,
            'total_reward': 0.0,
            'best_reward': -float('inf'),
            'best_molecule': None,
            'valid_molecules': 0,
            'unique_molecules': set()
        }

        logger.info(f" Improved RL Generator initialized for {target_protein}")

    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device"""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    async def calculate_reward(self, smiles: str) -> Dict[str, float]:
        """Calculate comprehensive reward for molecule"""

        # Check cache first
        if smiles in self.reward_cache:
            return self.reward_cache[smiles]

        rewards = {}

        # 1. Chemical validity
        mol = Chem.MolFromSmiles(smiles)
        
        # CRITICAL: Reject multi-fragment molecules (dots in SMILES)
        if mol is None or '.' in smiles:
            logger.debug(f"Invalid/fragmented SMILES: {smiles}")
            rewards = {
                'total_reward': -0.5,  # NEGATIVE reward for bad molecules!
                'binding_affinity': 0.0,
                'drug_likeness': 0.0,
                'molecular_properties': 0.0,
                'validity': 0.0
            }
            self.reward_cache[smiles] = rewards
            return rewards
        
        # Check molecule size constraints
        num_atoms = mol.GetNumHeavyAtoms()
        if num_atoms < 10 or num_atoms > 50:
            logger.debug(f"Molecule size out of range: {num_atoms} atoms")
            rewards = {
                'total_reward': -0.2,  # Penalty for wrong size
                'binding_affinity': 0.0,
                'drug_likeness': 0.1,
                'molecular_properties': 0.1,
                'validity': 0.5
            }
            self.reward_cache[smiles] = rewards
            return rewards

        rewards['validity'] = 1.0

        # 2. Drug-likeness (QED score)
        try:
            qed_score = QED.qed(mol)
            rewards['drug_likeness'] = qed_score
        except Exception:
            rewards['drug_likeness'] = 0.0

        # 3. Molecular properties (Lipinski-like)
        prop_score = self._calculate_property_score(mol)
        rewards['molecular_properties'] = prop_score

        # 4. Binding affinity: prefer QSAR (real ML prediction) > docking > simulated
        if self.has_qsar and self.target_protein in self.qsar_predictor.available_targets:
            binding_reward = self.qsar_predictor.get_binding_reward(smiles, self.target_protein)
            rewards['binding_source'] = 'QSAR'
        elif self.has_docking:
            binding_reward = await self._real_docking_reward(smiles)
            rewards['binding_source'] = 'docking'
        else:
            binding_reward = self._simulated_binding_reward(mol)
            rewards['binding_source'] = 'simulated'

        rewards['binding_affinity'] = binding_reward

        # 5. Synthetic Accessibility
        sa_score = self._calculate_sa_score(mol)
        rewards['synthetic_accessibility'] = sa_score

        # 6. ADMET prediction (NEW!)
        if self.has_admet:
            admet_score = self.admet_agent.get_admet_reward(smiles)
        else:
            admet_score = 0.5  # Default if not available
        rewards['admet_score'] = admet_score

        # 7. Chemical Safety Check (NEW!)
        if SAFETY_FILTER_AVAILABLE:
            is_safe, issues = self.safety_filter.check_safety(smiles)
            if not is_safe:
                # HEAVY PENALTY for reactive groups
                logger.debug(f"⚠️  Unsafe molecule: {smiles[:50]}... Issues: {', '.join(issues)}")
                rewards['safety'] = 0.0
                rewards['total_reward'] = -0.5  # Strong negative reward
                self.reward_cache[smiles] = rewards
                return rewards
            else:
                rewards['safety'] = 1.0
        else:
            rewards['safety'] = 1.0  # No filter available

        # 8. Calculate total reward with updated weights
        weights = {
            'binding_affinity': 0.30,  # Primary objective (reduced)
            'drug_likeness': 0.15,  # Drug-like properties (QED)
            'molecular_properties': 0.10,  # Lipinski properties
            'synthetic_accessibility': 0.10,  # Can it be synthesized?
            'admet_score': 0.15,  # ADMET safety profile
            'safety': 0.15,  # Chemical safety (NEW!)
            'validity': 0.05  # Must be valid
        }

        total_reward = sum(
            weights[key] * rewards[key]
            for key in weights.keys()
        )

        rewards['total_reward'] = total_reward

        # Debug logging for first few generations
        if len(self.reward_cache) < 10:
            logger.info(f"Reward breakdown for {smiles[:30]}...")
            logger.info(f" Binding: {rewards['binding_affinity']:.3f}")
            logger.info(f" Drug-like: {rewards['drug_likeness']:.3f}")
            logger.info(f" Properties: {rewards['molecular_properties']:.3f}")
            logger.info(f" Total: {total_reward:.3f}")

        # Cache result
        self.reward_cache[smiles] = rewards

        return rewards

    async def _real_docking_reward(self, smiles: str) -> float:
        """Get binding reward from real molecular docking"""
        try:
            result = await self.docking_agent.dock_molecule(
                smiles=smiles,
                target_protein=self.target_protein,
                generate_poses=3,
                optimize_geometry=True
            )

            # Convert binding affinity to reward (0-1 scale)
            affinity = result.binding_affinity
            
            # FIXED SCALING: Typical drug affinities are -5 to -12 kcal/mol
            # -5 kcal/mol -> 0.0 (weak binding)
            # -8.5 kcal/mol -> 0.5 (moderate binding)  
            # -12 kcal/mol -> 1.0 (strong binding)
            reward = max(0.0, min(1.0, (-affinity - 5.0) / 7.0))

            # Confidence adjustment (not bonus, just adjustment)
            confidence_factor = {
                "high": 1.0,
                "medium": 0.9,
                "low": 0.7
            }.get(result.confidence, 0.8)

            final_reward = reward * confidence_factor
            logger.debug(f"Docking: {smiles[:30]}... affinity={affinity:.2f} kcal/mol, reward={final_reward:.3f}")
            return final_reward

        except Exception as e:
            logger.warning(f"Real docking failed for {smiles}: {e}, using simulated reward")
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                return self._simulated_binding_reward(mol)
            return 0.1

    def _simulated_binding_reward(self, mol: Chem.Mol) -> float:
        """Simulate binding reward using molecular descriptors"""
        try:
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)

            # COX2-specific optimization (anti-inflammatory drugs)
            if self.target_protein == "COX2":
                mw_score = max(0, 1 - abs(mw - 350) / 150)
                logp_score = max(0, 1 - abs(logp - 3.0) / 2.0)
                tpsa_score = max(0, 1 - abs(tpsa - 50) / 40)
            else:
                mw_score = max(0, 1 - abs(mw - 400) / 200)
                logp_score = max(0, 1 - abs(logp - 2.5) / 2.5)
                tpsa_score = max(0, 1 - abs(tpsa - 60) / 50)

            # H-bonding capability
            hb_score = min(1.0, (hbd + hba) / 8.0)

            # Combine scores
            binding_score = (mw_score + logp_score + tpsa_score + hb_score) / 4.0

            return binding_score

        except Exception:
            return 0.0

    def _calculate_property_score(self, mol: Chem.Mol) -> float:
        """Calculate molecular property score"""
        try:
            violations = 0

            # Lipinski's Rule of 5
            if Descriptors.MolWt(mol) > 500:
                violations += 1
            if Descriptors.MolLogP(mol) > 5:
                violations += 1
            if Descriptors.NumHDonors(mol) > 5:
                violations += 1
            if Descriptors.NumHAcceptors(mol) > 10:
                violations += 1

            # Additional constraints
            if Descriptors.TPSA(mol) > 140:
                violations += 1
            if Descriptors.NumRotatableBonds(mol) > 10:
                violations += 1

            # Convert to score
            return max(0.0, (6 - violations) / 6.0)

        except Exception:
            return 0.0

    def _calculate_sa_score(self, mol: Chem.Mol) -> float:
        """Calculate Synthetic Accessibility score (normalized 0-1, higher = easier)"""
        try:
            if not SA_SCORE_AVAILABLE or sascorer is None:
                return 0.5  # Default if SA scorer not available
            
            # SA score ranges from 1 (easy) to 10 (hard)
            sa_raw = sascorer.calculateScore(mol)
            
            # Reject molecules with SA > 6 (too hard to synthesize)
            if sa_raw > 6:
                return 0.0  # Penalty for hard-to-synthesize
            
            # Normalize: 1 -> 1.0, 6 -> 0.0
            sa_normalized = max(0.0, (6 - sa_raw) / 5.0)
            return sa_normalized
            
        except Exception:
            return 0.3  # Default on error


    async def generate_and_evaluate_molecule(self) -> Dict[str, Any]:
        """Generate single molecule and evaluate"""

        # Choose generation strategy
        if self.use_neural:
            # Neural network generation with temperature-based exploration
            if random.random() < self.epsilon:
                # Exploration: High temperature for diversity
                temperature = random.uniform(1.2, 1.5)
            else:
                # Exploitation: Low temperature for refinement
                temperature = random.uniform(0.6, 0.9)
            
            smiles = self.smiles_generator.generate(
                temperature=temperature,
                max_length=100,
                top_p=0.9
            )
        else:
            # Fallback to scaffold-based generation
            if random.random() < self.epsilon:
                # Exploration: Random generation
                if random.random() < 0.5:
                    smiles = self.smiles_generator.generate_scaffold_molecule()
                else:
                    smiles = self.smiles_generator.generate_molecule_by_modification()
            else:
                # Exploitation: Modify best known molecule
                if self.stats['best_molecule']:
                    smiles = self.smiles_generator.generate_molecule_by_modification(
                        self.stats['best_molecule'], num_modifications=random.randint(1, 3)
                    )
                else:
                    smiles = self.smiles_generator.generate_scaffold_molecule()

        # Evaluate molecule
        rewards = await self.calculate_reward(smiles)

        # Update statistics
        self.stats['generation'] += 1
        self.stats['total_reward'] += rewards['total_reward']

        if rewards['validity'] > 0.5:
            self.stats['valid_molecules'] += 1
            self.stats['unique_molecules'].add(smiles)

        # Update best molecule
        if rewards['total_reward'] > self.stats['best_reward']:
            self.stats['best_reward'] = rewards['total_reward']
            self.stats['best_molecule'] = smiles
        
        # Train neural network on reward (policy gradient)
        if self.use_neural and rewards['validity'] > 0.5:
            self.smiles_generator.train_on_reward(smiles, rewards['total_reward'])

        return {
            'smiles': smiles,
            'rewards': rewards,
            'generation': self.stats['generation'],
            'is_best': rewards['total_reward'] == self.stats['best_reward']
        }

    async def train(self, num_generations: int = 100) -> List[Dict]:
        """Train the RL generator"""

        logger.info(f" Starting RL training: {num_generations} generations")

        training_history = []

        for gen in range(num_generations):
            result = await self.generate_and_evaluate_molecule()
            training_history.append(result)

            # Decay exploration rate
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            # Periodic logging
            if gen % 20 == 0:
                avg_reward = self.stats['total_reward'] / max(1, self.stats['generation'])
                logger.info(
                    f"Generation {gen}/{num_generations}: "
                    f"Best={self.stats['best_reward']:.3f}, "
                    f"Avg={avg_reward:.3f}, "
                    f"Valid={self.stats['valid_molecules']}, "
                    f"Unique={len(self.stats['unique_molecules'])}, "
                    f"ε={self.epsilon:.3f}"
                )

        logger.info(f" RL training complete!")
        logger.info(f" Best molecule: {self.stats['best_molecule']}")
        logger.info(f" Best reward: {self.stats['best_reward']:.3f}")
        logger.info(f" Valid molecules: {self.stats['valid_molecules']}")
        logger.info(f" Unique molecules: {len(self.stats['unique_molecules'])}")

        return training_history

    async def generate_optimized_library(self, library_size: int = 50) -> List[Dict]:
        """Generate library of optimized molecules"""

        logger.info(f" Generating optimized library: {library_size} molecules")

        # Set exploration to minimum for exploitation
        old_epsilon = self.epsilon
        self.epsilon = 0.01

        library = []

        for i in range(library_size):
            result = await self.generate_and_evaluate_molecule()
            library.append(result)

            if i % 10 == 0:
                logger.info(f"Generated {i + 1}/{library_size} molecules")

        # Restore epsilon
        self.epsilon = old_epsilon

        # Sort by reward
        library.sort(key=lambda x: x['rewards']['total_reward'], reverse=True)

        logger.info(f" Library generation complete!")
        logger.info(f" Best in library: {library[0]['rewards']['total_reward']:.3f}")

        return library


# Test the improved system
async def test_improved_rl_system():
    """Test the improved RL molecular generation system"""

    print(" Testing Improved RL Molecular Generation System...")
    print("=" * 60)

    # Initialize improved generator
    rl_gen = ImprovedRLMolecularGenerator(target_protein="COX2")

    print(f" System Status:")
    print(f" Target Protein: {rl_gen.target_protein}")
    print(f" Device: {rl_gen.device}")
    print(f" Real Docking: {rl_gen.has_docking}")
    print(f" Known Scaffolds: {len(rl_gen.smiles_generator.core_scaffolds)}")
    print(f" Functional Groups: {len(rl_gen.smiles_generator.functional_groups)}")

    # Test single generation
    print(f"\n Testing Single Molecule Generation:")
    print("-" * 40)

    for i in range(5):
        result = await rl_gen.generate_and_evaluate_molecule()

        print(f" {i + 1}. {result['smiles']}")
        print(f" Total Reward: {result['rewards']['total_reward']:.3f}")
        print(f" Binding: {result['rewards']['binding_affinity']:.3f}")
        print(f" Drug-like: {result['rewards']['drug_likeness']:.3f}")
        print(f" Valid: {result['rewards']['validity'] > 0.5}")
        print()

    # Test RL training
    print(f" Testing RL Training (25 generations):")
    print("-" * 40)

    training_history = await rl_gen.train(num_generations=25)

    # Show training progress
    print(f"\n Training Results:")
    print(f" Generations: {len(training_history)}")
    print(f" Best Reward: {rl_gen.stats['best_reward']:.3f}")
    print(f" Best Molecule: {rl_gen.stats['best_molecule']}")
    print(f" Valid Molecules: {rl_gen.stats['valid_molecules']}")
    print(f" Unique Molecules: {len(rl_gen.stats['unique_molecules'])}")

    # Generate optimized library
    print(f"\n Generating Optimized Molecule Library:")
    print("-" * 40)

    library = await rl_gen.generate_optimized_library(library_size=20)

    print(f"\n📚 Top 5 Optimized Molecules:")
    for i, mol in enumerate(library[:5]):
        print(f" {i + 1}. {mol['smiles']}")
        print(f" Reward: {mol['rewards']['total_reward']:.3f}")
        print(f" Binding: {mol['rewards']['binding_affinity']:.3f} kcal/mol")
        print(f" QED: {mol['rewards']['drug_likeness']:.3f}")

    print(f"\n Improved RL system test complete!")

    return training_history, library


if __name__ == "__main__":
    asyncio.run(test_improved_rl_system())
