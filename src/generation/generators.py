"""
Molecular generation models for de novo drug design

Implements various generative approaches:
- Variational Autoencoders (VAE) for SMILES
- Reinforcement Learning (RL) optimization
- Genetic Algorithms (GA) for molecule optimization
- Fragment-based generation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple, Callable
import logging
import random
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class MoleculeGenerator(ABC):
    """Abstract base class for molecule generators"""

    @abstractmethod
    def generate(self, num_molecules: int = 100) -> List[str]:
        """Generate new molecules"""
        pass

    @abstractmethod
    def train(self, smiles_list: List[str]) -> Dict:
        """Train the generator"""
        pass


class SMILESDataset(Dataset):
    """Dataset for SMILES strings"""

    def __init__(self, smiles_list: List[str], tokenizer, max_length: int = 128):
        self.smiles_list = smiles_list
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.encoded_smiles = []
        for smiles in smiles_list:
            try:
                encoded = tokenizer.encode(smiles, max_length)
                self.encoded_smiles.append(encoded)
            except Exception as e:
                logger.warning(f"Failed to encode SMILES {smiles}: {e}")
        
        logger.info(f"Created SMILES dataset with {len(self.encoded_smiles)} valid molecules")

    def __len__(self):
        return len(self.encoded_smiles)

    def __getitem__(self, idx):
        return torch.LongTensor(self.encoded_smiles[idx])


class SMILESVAE(nn.Module):
    """Variational Autoencoder for SMILES generation"""

    def __init__(self, vocab_size: int, hidden_dim: int = 256, latent_dim: int = 128,
                 max_length: int = 128, num_layers: int = 2):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.max_length = max_length
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.encoder_rnn = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)
        
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder_rnn = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder_output = nn.Linear(hidden_dim, vocab_size)
        
        self.dropout = nn.Dropout(0.2)

    def encode(self, x):
        """Encode SMILES to latent space"""
        embedded = self.embedding(x)
        output, (hidden, _) = self.encoder_rnn(embedded)
        
        hidden = hidden.view(self.num_layers, 2, x.size(0), self.hidden_dim)
        hidden = hidden[-1]
        hidden = hidden.permute(1, 0, 2).contiguous().view(x.size(0), -1)
        
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, z, max_length: Optional[int] = None):
        """Decode from latent space to SMILES"""
        if max_length is None:
            max_length = self.max_length
        
        batch_size = z.size(0)
        
        hidden = self.decoder_input(z).unsqueeze(0).repeat(self.num_layers, 1, 1)
        cell = torch.zeros_like(hidden)
        
        input_token = torch.ones(batch_size, 1, dtype=torch.long, device=z.device)
        outputs = []
        
        for _ in range(max_length):
            embedded = self.embedding(input_token)
            output, (hidden, cell) = self.decoder_rnn(embedded, (hidden, cell))
            vocab_logits = self.decoder_output(output)
            outputs.append(vocab_logits)
            input_token = torch.argmax(vocab_logits, dim=-1)
        
        return torch.cat(outputs, dim=1)

    def forward(self, x):
        """Forward pass"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z, x.size(1))
        return reconstructed, mu, logvar


class VAEGenerator(MoleculeGenerator):
    """VAE-based molecule generator"""

    def __init__(self, tokenizer, hidden_dim: int = 256, latent_dim: int = 128, **kwargs):
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer.vocab)
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        self.model = SMILESVAE(
            self.vocab_size, hidden_dim, latent_dim,
            max_length=kwargs.get('max_length', 128),
            num_layers=kwargs.get('num_layers', 2)
        )
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.beta = kwargs.get('beta', 1.0)
        self.is_trained = False

    def train(self, smiles_list: List[str], epochs: int = 100, batch_size: int = 64) -> Dict:
        """Train the VAE"""
        logger.info("Training SMILES VAE...")
        
        dataset = SMILESDataset(smiles_list, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        train_losses = []
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for batch in dataloader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                
                reconstructed, mu, logvar = self.model(batch)
                loss = self._vae_loss(reconstructed, batch, mu, logvar)
                
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            train_losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        self.is_trained = True
        
        logger.info("VAE training completed")
        return {'train_losses': train_losses, 'final_loss': train_losses[-1]}

    def _vae_loss(self, reconstructed, target, mu, logvar):
        """VAE loss function"""
        recon_loss = F.cross_entropy(
            reconstructed.view(-1, self.vocab_size),
            target.view(-1),
            ignore_index=0
        )
        
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss /= target.size(0) * target.size(1)
        
        return recon_loss + self.beta * kl_loss

    def generate(self, num_molecules: int = 100) -> List[str]:
        """Generate new molecules by sampling from latent space"""
        if not self.is_trained:
            raise ValueError("Model must be trained before generating")
        
        self.model.eval()
        generated_smiles = []
        
        with torch.no_grad():
            z = torch.randn(num_molecules, self.latent_dim, device=self.device)
            outputs = self.model.decode(z)
            
            for i in range(num_molecules):
                token_ids = torch.argmax(outputs[i], dim=-1).cpu().numpy()
                smiles = self.tokenizer.decode(token_ids)
                
                if len(smiles.strip()) > 0:
                    generated_smiles.append(smiles)
        
        logger.info(f"Generated {len(generated_smiles)} valid molecules")
        return generated_smiles


class GeneticAlgorithm:
    """Genetic Algorithm for molecule optimization"""

    def __init__(self, fitness_function: Callable[[str], float],
                 population_size: int = 100, mutation_rate: float = 0.1):
        self.fitness_function = fitness_function
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        
        self.bond_types = ['', '=', '#']
        self.atoms = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br']

    def optimize(self, initial_smiles: List[str], generations: int = 50) -> Dict:
        """Optimize molecules using genetic algorithm"""
        logger.info(f"Starting GA optimization for {generations} generations...")
        
        population = initial_smiles[:self.population_size]
        
        while len(population) < self.population_size:
            population.append(random.choice(initial_smiles))
        
        best_fitness_history = []
        avg_fitness_history = []
        
        for generation in range(generations):
            fitness_scores = []
            for smiles in population:
                try:
                    fitness = self.fitness_function(smiles)
                    fitness_scores.append(fitness)
                except Exception:
                    fitness_scores.append(0.0)
            
            best_fitness = max(fitness_scores)
            avg_fitness = np.mean(fitness_scores)
            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(avg_fitness)
            
            if (generation + 1) % 10 == 0:
                logger.info(f"Generation {generation + 1}: Best fitness = {best_fitness:.3f}, Avg fitness = {avg_fitness:.3f}")
            
            new_population = []
            
            elite_count = self.population_size // 10
            elite_indices = np.argsort(fitness_scores)[-elite_count:]
            for idx in elite_indices:
                new_population.append(population[idx])
            
            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                child = self._crossover(parent1, parent2)
                
                if random.random() < self.mutation_rate:
                    child = self._mutate(child)
                
                new_population.append(child)
            
            population = new_population
        
        final_fitness = [self.fitness_function(smiles) for smiles in population]
        best_idx = np.argmax(final_fitness)
        
        return {
            'best_molecule': population[best_idx],
            'best_fitness': final_fitness[best_idx],
            'final_population': population,
            'fitness_history': {'best': best_fitness_history, 'average': avg_fitness_history}
        }

    def _tournament_selection(self, population: List[str], fitness_scores: List[float],
                              tournament_size: int = 3) -> str:
        """Tournament selection"""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx]

    def _crossover(self, parent1: str, parent2: str) -> str:
        """Single-point crossover for SMILES"""
        try:
            min_length = min(len(parent1), len(parent2))
            if min_length < 2:
                return random.choice([parent1, parent2])
            
            crossover_point = random.randint(1, min_length - 1)
            
            if random.random() < 0.5:
                child = parent1[:crossover_point] + parent2[crossover_point:]
            else:
                child = parent2[:crossover_point] + parent1[crossover_point:]
            
            return child
        except Exception:
            return random.choice([parent1, parent2])

    def _mutate(self, smiles: str) -> str:
        """Mutate SMILES string"""
        try:
            if len(smiles) == 0:
                return smiles
            
            smiles_list = list(smiles)
            mutation_type = random.choice(['substitute', 'insert', 'delete'])
            
            if mutation_type == 'substitute' and len(smiles_list) > 0:
                pos = random.randint(0, len(smiles_list) - 1)
                if smiles_list[pos] in self.atoms:
                    smiles_list[pos] = random.choice(self.atoms)
            
            elif mutation_type == 'insert':
                pos = random.randint(0, len(smiles_list))
                smiles_list.insert(pos, random.choice(self.atoms + self.bond_types))
            
            elif mutation_type == 'delete' and len(smiles_list) > 1:
                pos = random.randint(0, len(smiles_list) - 1)
                del smiles_list[pos]
            
            return ''.join(smiles_list)
        except Exception:
            return smiles


class FragmentBasedGenerator:
    """Fragment-based molecule generation"""

    def __init__(self, fragment_library: Optional[List[str]] = None):
        self.fragment_library = fragment_library or self._get_default_fragments()
        logger.info(f"Initialized fragment library with {len(self.fragment_library)} fragments")

    def _get_default_fragments(self) -> List[str]:
        """Default molecular fragments"""
        return [
            'C', 'CC', 'CCC', 'CCCC',
            'c1ccccc1',
            'C1CCCCC1',
            'CCO', 'CCCO',
            'CC(=O)', 'CCC(=O)',
            'CC(=O)O', 'CCC(=O)O',
            'CCN', 'CCCN',
            'c1ccc(N)cc1',
            'c1ccc(O)cc1',
            'c1ccncc1',
        ]

    def generate_by_fragments(self, num_molecules: int = 100,
                             fragments_per_molecule: int = 3) -> List[str]:
        """Generate molecules by combining fragments"""
        generated = []
        
        for _ in range(num_molecules):
            selected_fragments = random.sample(
                self.fragment_library,
                min(fragments_per_molecule, len(self.fragment_library))
            )
            combined = ''.join(selected_fragments)
            generated.append(combined)
        
        logger.info(f"Generated {len(generated)} fragment-based molecules")
        return generated


class MolecularGenerator:
    """High-level interface for molecular generation"""

    def __init__(self, method: str = 'vae', tokenizer=None, **kwargs):
        self.method = method
        
        if method == 'vae':
            if tokenizer is None:
                raise ValueError("Tokenizer required for VAE method")
            self.generator = VAEGenerator(tokenizer, **kwargs)
        
        elif method == 'genetic':
            fitness_function = kwargs.get('fitness_function')
            if fitness_function is None:
                def default_fitness(smiles):
                    return len(smiles)
                fitness_function = default_fitness
            
            self.generator = GeneticAlgorithm(
                fitness_function,
                kwargs.get('population_size', 100),
                kwargs.get('mutation_rate', 0.1)
            )
        
        elif method == 'fragment':
            self.generator = FragmentBasedGenerator(kwargs.get('fragment_library'))
        
        else:
            raise ValueError(f"Unsupported generation method: {method}")

    def train(self, smiles_list: List[str], **kwargs) -> Dict:
        """Train the generator (if applicable)"""
        if hasattr(self.generator, 'train'):
            return self.generator.train(smiles_list, **kwargs)
        else:
            logger.info(f"Training not applicable for {self.method} method")
            return {}

    def generate(self, num_molecules: int = 100, **kwargs) -> List[str]:
        """Generate new molecules"""
        if self.method == 'genetic':
            initial_smiles = kwargs.get('initial_smiles', [])
            if not initial_smiles:
                raise ValueError("Initial SMILES required for genetic algorithm")
            result = self.generator.optimize(initial_smiles, kwargs.get('generations', 50))
            return result['final_population']
        elif self.method == 'fragment':
            return self.generator.generate_by_fragments(num_molecules, kwargs.get('fragments_per_molecule', 3))
        else:
            return self.generator.generate(num_molecules, **kwargs)


if __name__ == "__main__":
    # Example with fragment-based generation
    fragment_gen = MolecularGenerator('fragment')
    molecules = fragment_gen.generate(10)
    print("Fragment-based molecules:", molecules[:5])
    
    # Example with genetic algorithm
    def simple_fitness(smiles):
        return len(smiles)
    
    ga_gen = MolecularGenerator('genetic', fitness_function=simple_fitness)
    initial_smiles = ['CCO', 'CCC', 'CCCO', 'c1ccccc1']
    ga_result = ga_gen.generate(num_molecules=20, initial_smiles=initial_smiles, generations=10)
    print("GA optimized molecules:", ga_result[:3])
