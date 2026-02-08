#!/usr/bin/env python3
"""
Train Neural SMILES Generator on Large Dataset
Uses GPU acceleration for fast training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List
import logging
from tqdm import tqdm
import os

# Import our neural generator
from neural_smiles_generator import SMILESVocabulary, SMILESLSTM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SMILESDataset(Dataset):
    """Dataset for SMILES strings"""
    
    def __init__(self, smiles_file: str, vocab: SMILESVocabulary, max_length: int = 100):
        self.vocab = vocab
        self.max_length = max_length
        
        # Load SMILES
        logger.info(f"Loading SMILES from {smiles_file}...")
        with open(smiles_file, 'r') as f:
            self.smiles_list = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Loaded {len(self.smiles_list)} SMILES")
        
        # Preprocess all SMILES
        self.encoded_smiles = []
        for smiles in tqdm(self.smiles_list, desc="Encoding SMILES"):
            encoded = vocab.encode(smiles)
            if len(encoded) <= max_length:
                self.encoded_smiles.append(encoded)
        
        logger.info(f"Preprocessed {len(self.encoded_smiles)} valid sequences")
    
    def __len__(self):
        return len(self.encoded_smiles)
    
    def __getitem__(self, idx):
        encoded = self.encoded_smiles[idx]
        
        # Pad sequence
        padded = encoded + [self.vocab.pad_idx] * (self.max_length + 1 - len(encoded))
        padded = padded[:self.max_length + 1]
        
        # Input and target (shifted by 1)
        x = torch.tensor(padded[:-1], dtype=torch.long)
        y = torch.tensor(padded[1:], dtype=torch.long)
        
        return x, y


def train_smiles_generator(
    smiles_file: str,
    output_model_path: str = "smiles_lstm_pretrained.pt",
    batch_size: int = 128,
    epochs: int = 30,
    learning_rate: float = 0.001,
    device: str = "cuda"
):
    """
    Train LSTM on large SMILES dataset
    
    Args:
        smiles_file: Path to text file with SMILES (one per line)
        output_model_path: Where to save trained model
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Learning rate
        device: 'cuda' or 'cpu'
    """
    
    # Setup device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"
    
    device = torch.device(device)
    logger.info(f"Training on: {device}")
    
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Initialize vocabulary and model
    vocab = SMILESVocabulary()
    model = SMILESLSTM(
        vocab_size=vocab.vocab_size,
        embedding_dim=256,
        hidden_dim=512,
        num_layers=2,
        dropout=0.3
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load dataset
    dataset = SMILESDataset(smiles_file, vocab, max_length=100)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False
    )
    
    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Training loop
    logger.info(f"\nStarting training: {epochs} epochs, {len(dataset)} samples")
    logger.info("="*60)
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        # Progress bar
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            logits, _ = model(batch_x)
            
            # Calculate loss (ignore padding)
            loss = F.cross_entropy(
                logits.reshape(-1, vocab.vocab_size),
                batch_y.reshape(-1),
                ignore_index=vocab.pad_idx
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Track loss
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Epoch statistics
        avg_loss = total_loss / num_batches
        scheduler.step(avg_loss)
        
        logger.info(f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), output_model_path)
            logger.info(f"✓ Saved best model (loss: {best_loss:.4f})")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    logger.info("="*60)
    logger.info(f"✓ Training complete! Best loss: {best_loss:.4f}")
    logger.info(f"✓ Model saved to: {output_model_path}")
    
    # Test generation
    logger.info("\nTesting generation...")
    test_generation(model, vocab, device, num_samples=10)


def test_generation(model, vocab, device, num_samples=10):
    """Test the trained model by generating SMILES"""
    from rdkit import Chem
    
    model.eval()
    valid_count = 0
    
    logger.info("Generating test molecules...")
    
    with torch.no_grad():
        for i in range(num_samples):
            # Start with <START> token
            current_idx = torch.tensor([[vocab.start_idx]]).to(device)
            hidden = model.init_hidden(1, device)
            
            generated = []
            
            for _ in range(100):
                logits, hidden = model(current_idx, hidden)
                probs = F.softmax(logits[:, -1, :], dim=-1)
                
                # Sample
                next_idx = torch.multinomial(probs, 1)
                
                if next_idx.item() == vocab.end_idx:
                    break
                
                if next_idx.item() not in [vocab.pad_idx, vocab.start_idx]:
                    generated.append(next_idx.item())
                
                current_idx = next_idx
            
            smiles = vocab.decode(generated)
            mol = Chem.MolFromSmiles(smiles)
            
            valid = "✓" if mol else "✗"
            if mol:
                valid_count += 1
            
            logger.info(f"  {i+1}. {valid} {smiles}")
    
    logger.info(f"\nValidity: {valid_count}/{num_samples} ({valid_count/num_samples*100:.0f}%)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train SMILES LSTM")
    parser.add_argument("--data", type=str, default="chembl_100k.txt", help="SMILES data file")
    parser.add_argument("--output", type=str, default="smiles_lstm_pretrained.pt", help="Output model path")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    train_smiles_generator(
        smiles_file=args.data,
        output_model_path=args.output,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        device=args.device
    )
