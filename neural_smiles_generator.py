#!/usr/bin/env python3
"""
Neural SMILES Generator - LSTM-based character-level generation
Generates novel molecules through reinforcement learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import random
import os
from typing import List, Tuple, Optional
from rdkit import Chem
import logging

logger = logging.getLogger(__name__)


class SMILESVocabulary:
    """Vocabulary for SMILES character-level generation"""
    
    def __init__(self):
        # SMILES characters
        self.chars = [
            'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I',
            '(', ')', '[', ']', '=', '#', '@', '+', '-',
            '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'c', 'n', 'o', 's', 'p',  # Aromatic
            'H',  # Explicit hydrogen
            '<START>', '<END>', '<PAD>'
        ]
        
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        
        self.start_idx = self.char_to_idx['<START>']
        self.end_idx = self.char_to_idx['<END>']
        self.pad_idx = self.char_to_idx['<PAD>']
    
    def encode(self, smiles: str) -> List[int]:
        """Convert SMILES string to indices"""
        indices = [self.start_idx]
        i = 0
        while i < len(smiles):
            # Check for two-character tokens first
            if i < len(smiles) - 1 and smiles[i:i+2] in self.char_to_idx:
                indices.append(self.char_to_idx[smiles[i:i+2]])
                i += 2
            elif smiles[i] in self.char_to_idx:
                indices.append(self.char_to_idx[smiles[i]])
                i += 1
            else:
                # Skip unknown characters
                i += 1
        indices.append(self.end_idx)
        return indices
    
    def decode(self, indices: List[int]) -> str:
        """Convert indices back to SMILES string"""
        chars = []
        for idx in indices:
            if idx == self.end_idx:
                break
            if idx == self.start_idx or idx == self.pad_idx:
                continue
            if idx in self.idx_to_char:
                chars.append(self.idx_to_char[idx])
        return ''.join(chars)


class SMILESLSTM(nn.Module):
    """LSTM network for SMILES generation"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 256, 
                 hidden_dim: int = 512, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size)
        )
    
    def forward(self, x, hidden=None):
        """Forward pass"""
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        logits = self.fc(output)
        return logits, hidden
    
    def init_hidden(self, batch_size: int, device: torch.device):
        """Initialize hidden state"""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)


class NeuralSMILESGenerator:
    """Neural network-based SMILES generator with diversity mechanisms"""
    
    def __init__(self, device: str = "auto", pretrained_path: Optional[str] = None):
        """
        Initialize Neural SMILES Generator
        
        Args:
            device: Device to run on ('auto', 'cuda', 'cpu')
            pretrained_path: Optional path to pretrained model
        """
        self.device = self._setup_device(device)
        self.vocab = SMILESVocabulary()
        
        # Initialize LSTM model
        self.model = SMILESLSTM(
            vocab_size=self.vocab.vocab_size,
            embedding_dim=256,
            hidden_dim=512,
            num_layers=2,
            dropout=0.3
        ).to(self.device)
        
        # Initialize optimizer before pretraining
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # Load pretrained weights if available
        if pretrained_path:
            logger.info(f"Loading pretrained model from {pretrained_path}")
            self.model.load_state_dict(torch.load(pretrained_path, map_location=self.device))
        else:
            # Try to load the large pretrained model
            default_pretrained = "smiles_lstm_pretrained.pt"
            if os.path.exists(default_pretrained):
                logger.info(f"✓ Loading pretrained LSTM model (trained on 92K molecules)")
                self.model.load_state_dict(torch.load(default_pretrained, map_location=self.device))
            else:
                logger.warning(f"Pretrained model not found, using basic initialization")
                # Initialize with known drug SMILES for better starting point
                self._pretrain_on_drugs()
        
        # Diversity tracking
        self.generated_history = []
        self.max_history = 200
        
        # Valid molecule scaffolds as fallback
        self.fallback_scaffolds = [
            "c1ccccc1",  # Benzene
            "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
            "CC(C)Cc1ccc(C(C)C(=O)O)cc1",  # Ibuprofen
            "c1cncc2ccccc12",  # Quinoline
        ]
        
        logger.info(f"Neural SMILES Generator initialized on {self.device}")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device"""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def _pretrain_on_drugs(self):
        """Quick pretraining on known drug molecules"""
        known_drugs = [
            "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
            "CC(C)Cc1ccc(C(C)C(=O)O)cc1",  # Ibuprofen
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
            "COc1cc2[nH]c3c(c2cc1OC)CCN3C",  # Mescaline derivative
            "c1ccc(CCN)cc1",  # Phenethylamine
            "c1ccc(O)cc1",  # Phenol
            "c1ccc(N)cc1",  # Aniline
            "c1ccccc1",  # Benzene
            "c1ccc2ccccc2c1",  # Naphthalene
            "CCO",  # Ethanol
        ]
        
        logger.info("Pretraining on known drug molecules...")
        self.model.train()
        
        for epoch in range(5):  # Quick pretraining
            total_loss = 0
            for smiles in known_drugs:
                indices = self.vocab.encode(smiles)
                if len(indices) < 3:
                    continue
                
                # Create input/target pairs
                x = torch.tensor(indices[:-1]).unsqueeze(0).to(self.device)
                y = torch.tensor(indices[1:]).unsqueeze(0).to(self.device)
                
                # Forward pass
                logits, _ = self.model(x)
                
                # Calculate loss
                loss = F.cross_entropy(
                    logits.reshape(-1, self.vocab.vocab_size),
                    y.reshape(-1)
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if epoch == 0 or (epoch + 1) % 2 == 0:
                logger.info(f"Pretrain epoch {epoch+1}/5, loss: {total_loss/len(known_drugs):.4f}")
        
        self.model.eval()
        logger.info("Pretraining complete!")
    
    def generate(self, temperature: float = 1.0, max_length: int = 100, 
                 top_p: float = 0.9, max_attempts: int = 10) -> str:
        """
        Generate a SMILES string using the neural network
        
        Args:
            temperature: Sampling temperature (higher = more diverse)
            max_length: Maximum SMILES length
            top_p: Nucleus sampling threshold
            max_attempts: Max attempts to generate valid SMILES
        
        Returns:
            Valid SMILES string
        """
        self.model.eval()
        
        for attempt in range(max_attempts):
            try:
                smiles = self._generate_single(temperature, max_length, top_p)
                
                # Validate SMILES
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None and '.' not in smiles:  # Valid and single molecule
                    num_atoms = mol.GetNumHeavyAtoms()
                    if 5 <= num_atoms <= 60:  # Reasonable size
                        # Check diversity
                        if self._is_diverse_enough(smiles):
                            self.generated_history.append(smiles)
                            if len(self.generated_history) > self.max_history:
                                self.generated_history.pop(0)
                            return smiles
            
            except Exception as e:
                logger.debug(f"Generation attempt {attempt + 1} failed: {e}")
                continue
        
        # Fallback to scaffold if all attempts fail
        logger.warning(f"Failed to generate valid SMILES after {max_attempts} attempts, using fallback")
        return random.choice(self.fallback_scaffolds)
    
    def _generate_single(self, temperature: float, max_length: int, top_p: float) -> str:
        """Generate a single SMILES string"""
        with torch.no_grad():
            # Start with <START> token
            current_idx = torch.tensor([[self.vocab.start_idx]]).to(self.device)
            hidden = self.model.init_hidden(1, self.device)
            
            generated_indices = []
            
            for _ in range(max_length):
                # Forward pass
                logits, hidden = self.model(current_idx, hidden)
                logits = logits[:, -1, :] / temperature
                
                # Apply nucleus (top-p) sampling
                probs = F.softmax(logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumsum_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                
                # Set probabilities to zero
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                probs[:, indices_to_remove] = 0
                probs = probs / probs.sum()  # Renormalize
                
                # Sample next character
                dist = Categorical(probs)
                next_idx = dist.sample()
                
                # Check for end token
                if next_idx.item() == self.vocab.end_idx:
                    break
                
                # Skip pad and start tokens
                if next_idx.item() not in [self.vocab.pad_idx, self.vocab.start_idx]:
                    generated_indices.append(next_idx.item())
                
                current_idx = next_idx.unsqueeze(0)
            
            # Decode to SMILES
            smiles = self.vocab.decode(generated_indices)
            return smiles
    
    def _is_diverse_enough(self, smiles: str, similarity_threshold: float = 0.85) -> bool:
        """Check if molecule is diverse enough compared to recent history"""
        if not self.generated_history:
            return True
        
        try:
            from rdkit import DataStructs
            from rdkit.Chem import rdFingerprintGenerator
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            
            mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
            fp_new = mfpgen.GetFingerprint(mol)
            
            # Check against recent molecules
            for historical_smiles in self.generated_history[-50:]:
                historical_mol = Chem.MolFromSmiles(historical_smiles)
                if historical_mol is None:
                    continue
                
                fp_hist = mfpgen.GetFingerprint(historical_mol)
                similarity = DataStructs.TanimotoSimilarity(fp_new, fp_hist)
                
                if similarity > similarity_threshold:
                    return False  # Too similar
            
            return True
        
        except Exception:
            # If fingerprint calculation fails, allow the molecule
            return True
    
    def train_on_reward(self, smiles: str, reward: float):
        """Train the network using reward signal (policy gradient)"""
        try:
            self.model.train()
            
            # Encode SMILES
            indices = self.vocab.encode(smiles)
            if len(indices) < 3:
                return
            
            x = torch.tensor(indices[:-1]).unsqueeze(0).to(self.device)
            y = torch.tensor(indices[1:]).unsqueeze(0).to(self.device)
            
            # Forward pass
            logits, _ = self.model(x)
            
            # Calculate log probabilities
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Get log prob for actual tokens
            token_log_probs = log_probs.gather(2, y.unsqueeze(-1)).squeeze(-1)
            
            # Policy gradient loss: -log_prob * reward
            # Higher reward = reinforce this sequence
            # Use reward as advantage (simplified, no baseline)
            advantage = reward - 0.5  # 0.5 as simple baseline
            loss = -(token_log_probs.mean() * advantage)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            self.model.eval()
            
        except Exception as e:
            logger.debug(f"Training update failed: {e}")
    
    def save_model(self, path: str):
        """Save model weights"""
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model weights"""
        self.model.load_state_dict(torch.load(path))
        logger.info(f"Model loaded from {path}")


# Test the generator
if __name__ == "__main__":
    print("Testing Neural SMILES Generator...")
    
    generator = NeuralSMILESGenerator()
    
    print("\n" + "="*60)
    print("Generating molecules with different temperatures:")
    print("="*60)
    
    temps = [0.6, 0.8, 1.0, 1.2, 1.5]
    
    for temp in temps:
        print(f"\nTemperature: {temp}")
        molecules = []
        for i in range(5):
            smiles = generator.generate(temperature=temp, max_attempts=3)
            mol = Chem.MolFromSmiles(smiles)
            valid = "✓" if mol else "✗"
            molecules.append(smiles)
            print(f"  {i+1}. {valid} {smiles}")
        
        unique = len(set(molecules))
        print(f"  → Unique: {unique}/5 ({unique/5*100:.0f}%)")
    
    print("\n" + "="*60)
    print("Testing diversity mechanism:")
    print("="*60)
    
    # Generate 10 molecules with moderate temperature
    molecules = set()
    for i in range(10):
        smiles = generator.generate(temperature=1.0)
        molecules.add(smiles)
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            print(f"{i+1}. {smiles} ({mol.GetNumHeavyAtoms()} atoms)")
    
    print(f"\n✓ Generated {len(molecules)}/10 unique molecules ({len(molecules)/10*100:.0f}% diversity)")
    print("\nNeural SMILES Generator test complete!")
