#!/usr/bin/env python3
"""
REINFORCEMENT LEARNING MOLECULAR GENERATION AGENT

This implements state-of-the-art RL-based molecular generation using:
    - Policy Gradient Methods (REINFORCE)
    - Actor-Critic Networks
    - SMILES-based molecular representation
    - Real docking-based reward functions
    - Multi-objective optimization (binding affinity + drug-likeness)

Addresses Gap: Static molecular generation vs adaptive RL optimization
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
from torch.distributions import Categorical
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import asyncio
import random
from collections import deque
import pickle

# RDKit for molecular operations
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, QED
from rdkit.Chem.Scaffolds import MurckoScaffold

# Import our docking system
try:
    from real_docking_agent import RealMolecularDockingAgent
    DOCKING_AVAILABLE = True
except ImportError:
    DOCKING_AVAILABLE = False
    print(" Real docking not available - using simulated rewards")

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result from RL molecular generation"""
    smiles: str
    reward: float
    binding_affinity: float
    drug_likeness: float
    validity: bool
    uniqueness: float
    diversity: float
    generation_step: int
    target_protein: str


class SMILESVocabulary:
    """Vocabulary for SMILES tokenization"""

    def __init__(self):
        # Standard SMILES tokens
        self.tokens = [
            'PAD', 'START', 'END', 'UNK',  # Special tokens
            'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I',  # Atoms
            'c', 'n', 'o', 's', 'p',  # Aromatic atoms
            '(', ')', '[', ']',  # Brackets
            '=', '#', '-',  # Bonds
            '1', '2', '3', '4', '5', '6', '7', '8', '9',  # Ring numbers
            '+', '-', '@',  # Charges and chirality
            'H',  # Hydrogen
        ]

        # Create mappings
        self.token_to_idx = {token: idx for idx, token in enumerate(self.tokens)}
        self.idx_to_token = {idx: token for idx, token in enumerate(self.tokens)}
        self.vocab_size = len(self.tokens)

        # Special indices
        self.pad_idx = self.token_to_idx['PAD']
        self.start_idx = self.token_to_idx['START']
        self.end_idx = self.token_to_idx['END']
        self.unk_idx = self.token_to_idx['UNK']

    def tokenize(self, smiles: str, max_length: int = 100) -> List[int]:
        """Convert SMILES to token indices"""
        tokens = [self.start_idx]

        i = 0
        while i < len(smiles) and len(tokens) < max_length - 1:
            # Multi-character tokens first
            if i < len(smiles) - 1:
                two_char = smiles[i:i + 2]
                if two_char in self.token_to_idx:
                    tokens.append(self.token_to_idx[two_char])
                    i += 2
                    continue

            # Single character tokens
            char = smiles[i]
            if char in self.token_to_idx:
                tokens.append(self.token_to_idx[char])
            else:
                tokens.append(self.unk_idx)
            i += 1

        tokens.append(self.end_idx)

        # Pad to max length
        while len(tokens) < max_length:
            tokens.append(self.pad_idx)

        return tokens[:max_length]

    def detokenize(self, token_indices: List[int]) -> str:
        """Convert token indices back to SMILES"""
        tokens = []
        for idx in token_indices:
            if idx == self.end_idx:
                break
            if idx not in [self.pad_idx, self.start_idx]:
                tokens.append(self.idx_to_token.get(idx, 'UNK'))

        return ''.join(tokens)


class MolecularGeneratorNetwork(nn.Module):
    """
    Neural network for molecular generation using LSTM + attention
    """

    def __init__(
            self,
            vocab_size: int,
            hidden_size: int = 256,
            num_layers: int = 3,
            dropout: float = 0.2):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)

        # LSTM backbone
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            hidden_size, 8, dropout=dropout, batch_first=True)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, vocab_size)
        )

        # Value head for critic
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(
            self, x: torch.Tensor, hidden: Optional[Tuple] = None) -> Tuple[torch.Tensor, torch.Tensor, Tuple]:
        """
        Forward pass

        Args:
            x: Input token indices [batch_size, seq_len]
            hidden: Hidden state from previous step

        Returns:
            logits: Output logits [batch_size, seq_len, vocab_size]
            value: State value [batch_size, seq_len, 1]
            hidden: New hidden state
        """
        # Embedding
        embedded = self.embedding(x)  # [batch_size, seq_len, hidden_size]

        # LSTM
        lstm_out, hidden = self.lstm(embedded, hidden)  # [batch_size, seq_len, hidden_size]

        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Residual connection
        output = lstm_out + attn_out

        # Output projections
        logits = self.output_proj(output)  # [batch_size, seq_len, vocab_size]
        value = self.value_head(output)  # [batch_size, seq_len, 1]

        return logits, value.squeeze(-1), hidden


class MolecularRewardFunction:
    """
    Multi-objective reward function for molecular generation
    """

    def __init__(self, target_protein: str = "COX2", weights: Optional[Dict[str, float]] = None):
        self.target_protein = target_protein

        # Default reward weights
        self.weights = weights or {
            'binding_affinity': 0.4,  # Real docking score
            'drug_likeness': 0.25,  # QED score
            'lipinski': 0.15,  # Rule of 5 compliance
            'novelty': 0.1,  # Structural novelty
            'validity': 0.1  # Chemical validity
        }

        # Initialize docking engine
        if DOCKING_AVAILABLE:
            self.docking_agent = RealMolecularDockingAgent()
            self.has_docking = True
        else:
            self.docking_agent = None
            self.has_docking = False

        # Cache for expensive calculations
        self.docking_cache = {}
        self.known_molecules = set()

    async def calculate_reward(self, smiles: str, generation_step: int = 0) -> Dict[str, float]:
        """Calculate comprehensive reward for generated molecule"""
        rewards = {}

        # 1. Chemical validity
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {
                'total_reward': -10.0,
                'binding_affinity': -10.0,
                'drug_likeness': 0.0,
                'lipinski': 0.0,
                'novelty': 0.0,
                'validity': 0.0
            }

        rewards['validity'] = 1.0

        # 2. Drug-likeness (QED)
        try:
            qed_score = QED.qed(mol)
            rewards['drug_likeness'] = qed_score
        except:
            rewards['drug_likeness'] = 0.0

        # 3. Lipinski's Rule of 5
        lipinski_score = self._calculate_lipinski_score(mol)
        rewards['lipinski'] = lipinski_score

        # 4. Novelty (based on known molecules)
        novelty_score = self._calculate_novelty(smiles)
        rewards['novelty'] = novelty_score

        # 5. Binding affinity (real docking or simulation)
        if self.has_docking:
            binding_reward = await self._calculate_binding_reward_real(smiles)
        else:
            binding_reward = self._calculate_binding_reward_simulated(mol)

        rewards['binding_affinity'] = binding_reward

        # Calculate weighted total reward
        total_reward = sum(
            self.weights[key] * rewards[key]
            for key in self.weights.keys()
        )

        rewards['total_reward'] = total_reward

        return rewards

    async def _calculate_binding_reward_real(self, smiles: str) -> float:
        """Calculate binding reward using real docking"""
        cache_key = f"{smiles}_{self.target_protein}"

        if cache_key in self.docking_cache:
            return self.docking_cache[cache_key]

        try:
            # Perform real docking
            result = await self.docking_agent.dock_molecule(
                smiles=smiles,
                target_protein=self.target_protein,
                generate_poses=3,  # Fast generation
                optimize_geometry=True
            )

            # Convert binding affinity to reward (0-1 scale)
            # Strong binding: -10 kcal/mol = 1.0, Weak binding: 0 kcal/mol = 0.0
            affinity = result.binding_affinity
            reward = max(0.0, min(1.0, (-affinity) / 10.0))

            # Bonus for high confidence
            if result.confidence == "high":
                reward *= 1.2
            elif result.confidence == "low":
                reward *= 0.8

            reward = min(1.0, reward)  # Clamp to [0,1]

            self.docking_cache[cache_key] = reward
            return reward

        except Exception as e:
            logger.warning(f"Real docking failed for {smiles}: {e}")
            return 0.0

    def _calculate_binding_reward_simulated(self, mol: Chem.Mol) -> float:
        """Simulate binding reward using molecular descriptors"""
        try:
            # Use molecular descriptors as proxy for binding
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)

            # Simulate optimal ranges for binding
            mw_score = max(0, 1 - abs(mw - 400) / 200)  # Optimal ~400 Da
            logp_score = max(0, 1 - abs(logp - 2.5) / 2.5)  # Optimal ~2.5
            tpsa_score = max(0, 1 - abs(tpsa - 60) / 60)  # Optimal ~60 Ų

            # H-bonding potential
            hb_score = min(1.0, (hbd + hba) / 8.0)

            # Combine scores
            simulated_binding = (mw_score + logp_score + tpsa_score + hb_score) / 4.0

            return simulated_binding

        except:
            return 0.0

    def _calculate_lipinski_score(self, mol: Chem.Mol) -> float:
        """Calculate Lipinski Rule of 5 compliance score"""
        try:
            violations = 0

            # Rule 1: MW <= 500
            if Descriptors.MolWt(mol) > 500:
                violations += 1

            # Rule 2: LogP <= 5
            if Descriptors.MolLogP(mol) > 5:
                violations += 1

            # Rule 3: HBD <= 5
            if Descriptors.NumHDonors(mol) > 5:
                violations += 1

            # Rule 4: HBA <= 10
            if Descriptors.NumHAcceptors(mol) > 10:
                violations += 1

            # Convert violations to score
            return (4 - violations) / 4.0

        except:
            return 0.0

    def _calculate_novelty(self, smiles: str) -> float:
        """Calculate structural novelty score"""
        if smiles in self.known_molecules:
            return 0.0

        # Add to known molecules
        self.known_molecules.add(smiles)

        # Simple novelty based on complexity
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0

        # Novelty based on ring count and complexity
        ring_count = Descriptors.RingCount(mol)
        heavy_atoms = mol.GetNumHeavyAtoms()

        # Higher complexity = higher novelty (up to a point)
        complexity_score = min(1.0, (ring_count + heavy_atoms / 20) / 3.0)

        return complexity_score


class RLMolecularGenerator:
    """
    Reinforcement Learning-based Molecular Generator

    Uses policy gradient methods to generate molecules optimized for:
        - Target protein binding (via real docking)
        - Drug-likeness properties
        - Chemical novelty and diversity
    """

    def __init__(
            self,
            target_protein: str = "COX2",
            hidden_size: int = 256,
            learning_rate: float = 1e-4,
            device: str = "auto"
    ):
        self.target_protein = target_protein
        self.device = self._setup_device(device)

        # Initialize vocabulary
        self.vocab = SMILESVocabulary()

        # Initialize networks
        self.generator = MolecularGeneratorNetwork(
            vocab_size=self.vocab.vocab_size,
            hidden_size=hidden_size
        ).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.generator.parameters(), lr=learning_rate)

        # Reward function
        self.reward_function = MolecularRewardFunction(target_protein)

        # Training memory
        self.memory = deque(maxlen=10000)
        self.generated_molecules = set()

        # Training statistics
        self.stats = {
            'episodes': 0,
            'total_reward': 0.0,
            'best_reward': -float('inf'),
            'best_molecule': None,
            'valid_molecules': 0,
            'unique_molecules': 0
        }

        logger.info(f" RL Molecular Generator initialized for {target_protein}")

    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device"""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        return torch.device(device)

    async def generate_molecule(self, max_length: int = 100, temperature: float = 1.0) -> GenerationResult:
        """Generate a single molecule using current policy"""

        self.generator.eval()

        with torch.no_grad():
            # Initialize generation
            sequence = [self.vocab.start_idx]
            hidden = None
            log_probs = []

            # Generate tokens one by one
            for step in range(max_length - 1):
                # Current input
                input_tokens = torch.tensor([sequence], dtype=torch.long, device=self.device)

                # Forward pass
                logits, value, hidden = self.generator(input_tokens, hidden)

                # Get logits for next token
                next_logits = logits[0, -1, :] / temperature

                # Sample next token
                probs = F.softmax(next_logits, dim=-1)
                dist = Categorical(probs)
                next_token = dist.sample()

                # Store log probability
                log_probs.append(dist.log_prob(next_token))

                # Add to sequence
                sequence.append(next_token.item())

                # Stop if END token
                if next_token.item() == self.vocab.end_idx:
                    break

            # Convert to SMILES
            smiles = self.vocab.detokenize(sequence)

            # Calculate rewards
            rewards = await self.reward_function.calculate_reward(smiles, self.stats['episodes'])

            # Create result
            result = GenerationResult(
                smiles=smiles,
                reward=rewards['total_reward'],
                binding_affinity=rewards['binding_affinity'],
                drug_likeness=rewards['drug_likeness'],
                validity=rewards['validity'] > 0.5,
                uniqueness=1.0 if smiles not in self.generated_molecules else 0.0,
                diversity=rewards['novelty'],
                generation_step=self.stats['episodes'],
                target_protein=self.target_protein
            )

            # Update tracking
            if result.validity:
                self.generated_molecules.add(smiles)

            return result

    async def train_episode(self, num_generations: int = 10) -> Dict[str, float]:
        """Train for one episode using REINFORCE algorithm"""

        self.generator.train()
        episode_rewards = []
        episode_molecules = []

        # Generate molecules for this episode
        for _ in range(num_generations):
            result = await self.generate_molecule()
            episode_rewards.append(result.reward)
            episode_molecules.append(result)

            # Store in memory
            self.memory.append(result)

        # Calculate baseline (running average)
        baseline = np.mean([r.reward for r in self.memory]) if self.memory else 0.0

        # Policy gradient update
        total_loss = 0.0
        self.optimizer.zero_grad()

        for result in episode_molecules:
            if result.validity:
                # Calculate advantage
                advantage = result.reward - baseline

                # Generate sequence for gradient calculation
                sequence = self.vocab.tokenize(result.smiles)
                input_tokens = torch.tensor([sequence[:-1]], dtype=torch.long, device=self.device)
                target_tokens = torch.tensor([sequence[1:]], dtype=torch.long, device=self.device)

                # Forward pass
                logits, value, _ = self.generator(input_tokens)

                # Calculate policy loss
                log_probs = F.log_softmax(logits, dim=-1)
                selected_log_probs = log_probs.gather(2, target_tokens.unsqueeze(2)).squeeze(2)

                # REINFORCE loss
                policy_loss = -(selected_log_probs.mean() * advantage)

                # Value loss (critic)
                value_loss = F.mse_loss(value.mean(), torch.tensor(result.reward, device=self.device))

                # Combined loss
                loss = policy_loss + 0.5 * value_loss
                total_loss += loss.item()

                loss.backward()

        # Update parameters
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update statistics
        episode_reward = np.mean(episode_rewards)
        self.stats['episodes'] += 1
        self.stats['total_reward'] += episode_reward
        self.stats['valid_molecules'] += sum(1 for r in episode_molecules if r.validity)
        self.stats['unique_molecules'] = len(self.generated_molecules)

        # Track best molecule
        best_result = max(episode_molecules, key=lambda x: x.reward)
        if best_result.reward > self.stats['best_reward']:
            self.stats['best_reward'] = best_result.reward
            self.stats['best_molecule'] = best_result

        return {
            'episode_reward': episode_reward,
            'total_loss': total_loss,
            'baseline': baseline,
            'best_reward': self.stats['best_reward'],
            'valid_molecules': self.stats['valid_molecules'],
            'unique_molecules': self.stats['unique_molecules']
        }

    async def train(self, num_episodes: int = 100, generations_per_episode: int = 10) -> List[Dict]:
        """Train the RL generator"""

        logger.info(f" Starting RL training: {num_episodes} episodes, {generations_per_episode} generations each")

        training_history = []

        for episode in range(num_episodes):
            # Train episode
            episode_stats = await self.train_episode(generations_per_episode)
            training_history.append(episode_stats)

            # Logging
            if episode % 10 == 0:
                logger.info(f"Episode {episode}/{num_episodes}: "
                            f"Reward={episode_stats['episode_reward']:.3f}, "
                            f"Valid={episode_stats['valid_molecules']}, "
                            f"Unique={episode_stats['unique_molecules']}")

        logger.info(f" RL training complete! Best molecule: {self.stats['best_molecule'].smiles if self.stats['best_molecule'] else 'None'}")

        return training_history

    async def generate_optimized_molecules(self, num_molecules: int = 50, temperature: float = 0.8) -> List[GenerationResult]:
        """Generate optimized molecules using trained policy"""

        logger.info(f" Generating {num_molecules} optimized molecules")

        molecules = []

        for i in range(num_molecules):
            result = await self.generate_molecule(temperature=temperature)
            molecules.append(result)

            if i % 10 == 0:
                logger.info(f"Generated {i+1}/{num_molecules} molecules")

        # Sort by reward
        molecules.sort(key=lambda x: x.reward, reverse=True)

        logger.info(f" Generation complete! Best reward: {molecules[0].reward:.3f}")

        return molecules

    def save_model(self, filepath: str):
        """Save trained model"""
        checkpoint = {
            'model_state_dict': self.generator.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'vocab': self.vocab,
            'stats': self.stats,
            'target_protein': self.target_protein
        }

        torch.save(checkpoint, filepath)
        logger.info(f" Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.generator.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.vocab = checkpoint['vocab']
        self.stats = checkpoint['stats']
        self.target_protein = checkpoint['target_protein']

        logger.info(f"📂 Model loaded from {filepath}")


# Test function
async def test_rl_molecular_generation():
    """Test the RL molecular generation system"""

    print(" Testing RL Molecular Generation System...")

    # Initialize generator
    rl_generator = RLMolecularGenerator(target_protein="COX2", hidden_size=128)

    print(f" Generator initialized:")
    print(f" Target Protein: {rl_generator.target_protein}")
    print(f" Vocabulary Size: {rl_generator.vocab.vocab_size}")
    print(f" Device: {rl_generator.device}")
    print(f" Real Docking: {DOCKING_AVAILABLE}")

    # Test single generation
    print(f"\n Testing single molecule generation...")

    result = await rl_generator.generate_molecule()

    print(f" Generated: {result.smiles}")
    print(f" Total Reward: {result.reward:.3f}")
    print(f" Binding Affinity: {result.binding_affinity:.3f}")
    print(f" Drug-likeness: {result.drug_likeness:.3f}")
    print(f" Validity: {result.validity}")

    # Test short training
    print(f"\n Testing RL training (5 episodes)...")

    training_history = await rl_generator.train(num_episodes=5, generations_per_episode=5)

    print(f" Training Episodes: {len(training_history)}")
    print(f" Final Reward: {training_history[-1]['episode_reward']:.3f}")
    print(f" Best Reward: {training_history[-1]['best_reward']:.3f}")
    print(f" Valid Molecules: {training_history[-1]['valid_molecules']}")

    # Generate optimized molecules
    print(f"\n Testing optimized generation...")

    optimized_molecules = await rl_generator.generate_optimized_molecules(num_molecules=10)

    print(f" Generated Molecules: {len(optimized_molecules)}")
    print(f" Top 3 molecules:")

    for i, mol in enumerate(optimized_molecules[:3]):
        print(f" {i+1}. {mol.smiles} (Reward: {mol.reward:.3f})")

    print("\n RL Molecular Generation test complete!")


if __name__ == "__main__":
    asyncio.run(test_rl_molecular_generation())