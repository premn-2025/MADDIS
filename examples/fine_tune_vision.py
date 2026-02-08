#!/usr/bin/env python3
"""
Fine-Tune Vision Models for Molecular Property Prediction

Example script showing how to fine-tune popular vision models on molecular images:
    - ResNet50/101 (Residual Networks)
    - EfficientNet (Efficient scaling)
    - Vision Transformer (ViT) (Attention-based)

    Optimized for RTX 4060 with mixed precision and gradient accumulation.

    Usage:
        python examples/fine_tune_vision.py --model resnet50 --property logp --epochs 20

        Author: AI Drug Discovery Pipeline
        """

import os
import sys
import argparse
    import time
    import json
    from pathlib import Path
    from typing import Dict, Any

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.cuda.amp import autocast, GradScaler
    from torch.utils.tensorboard import SummaryWriter
    import torchvision.models as models
    import numpy as np
    from tqdm import tqdm

# Add src to path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from src.data.fine_tuning_prep import FineTuningDataPreparer, FineTuningConfig

    class MolecularPropertyPredictor(nn.Module):
        """
        Vision model adapted for molecular property prediction

        Takes pre-trained vision models and adapts them for:
            - Single property regression (e.g., LogP, MW)
            - Multi-property regression (multiple outputs)
            - Binary classification (drug-like vs non-drug-like)
            """

        def __init__(self,
                    model_name: str = "resnet50",
                    num_properties: int = 1,
                    pretrained: bool = True,
                    dropout: float = 0.3):
            """
                Args:
                    model_name: Name of backbone model
                    num_properties: Number of properties to predict
                    pretrained: Use ImageNet pretrained weights
                    dropout: Dropout rate for regularization
                    """
                super().__init__()

                self.model_name = model_name
                self.num_properties = num_properties

# Load backbone model
                    if model_name == "resnet50":
                        self.backbone = models.resnet50(pretrained=pretrained)
                        feature_dim = self.backbone.fc.in_features
                        self.backbone.fc = nn.Identity()  # Remove final layer

                    elif model_name == "resnet101":
                        self.backbone = models.resnet101(pretrained=pretrained)
                        feature_dim = self.backbone.fc.in_features
                        self.backbone.fc = nn.Identity()

                    elif model_name == "efficientnet_b0":
                        self.backbone = models.efficientnet_b0(
                            pretrained=pretrained)
                        feature_dim = self.backbone.classifier[1].in_features
                        self.backbone.classifier = nn.Identity()

                    elif model_name == "efficientnet_b3":
                        self.backbone = models.efficientnet_b3(
                            pretrained=pretrained)
                        feature_dim = self.backbone.classifier[1].in_features
                        self.backbone.classifier = nn.Identity()

                    elif model_name == "vit_b_16":
                        self.backbone = models.vit_b_16(pretrained=pretrained)
                        feature_dim = self.backbone.heads.head.in_features
                        self.backbone.heads.head = nn.Identity()

                    else:
                        raise ValueError(f"Unsupported model: {model_name}")

# Property prediction head
                    self.property_head = nn.Sequential(
                        nn.Dropout(dropout),
                        nn.Linear(feature_dim, 512),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(512, 256),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(256, num_properties)
                    )

                    def forward(self, x):
                        """Forward pass"""
                        features = self.backbone(x)
                        properties = self.property_head(features)
                        return properties

                    class MolecularVisionTrainer:
                        """
                        ‍♂ Trainer class for molecular vision models

                        Features:
                            - Mixed precision training (RTX 4060 optimization)
                            - Gradient accumulation for larger effective batch sizes
                            - Learning rate scheduling
                            - Early stopping
                            - TensorBoard logging
                            """

                        def __init__(self,
                                    model: nn.Module,
                                    train_loader: torch.utils.data.DataLoader,
                                    val_loader: torch.utils.data.DataLoader,
                                    property_names: list,
                                    device: torch.device,
                                    config: Dict[str, Any]):

                            self.model = model.to(device)
                            self.train_loader = train_loader
                                self.val_loader = val_loader
                                self.property_names = property_names
                                self.device = device
                                self.config = config

# Setup optimizer
                                self.optimizer = optim.AdamW(
                                    model.parameters(),
                                    lr=config['learning_rate'],
                                    weight_decay=config['weight_decay']
                                )

# Setup learning rate scheduler
                                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                                    self.optimizer,
                                    mode='min',
                                    factor=0.5,
                                    patience=3,
                                    verbose=True
                                )

# Setup loss function
                                self.criterion = nn.MSELoss()

# Mixed precision scaler for RTX 4060
                                self.scaler = GradScaler()

# Training tracking
                                self.best_val_loss = float('inf')
                                self.patience_counter = 0

# Logging
                                self.writer = SummaryWriter(config['log_dir'])

                                def train_epoch(
                                        self, epoch: int) -> Dict[str, float]:
                                    """Train for one epoch"""
                                    self.model.train()

                                    total_loss = 0
                                    num_batches = 0

                                    progress_bar = tqdm(
                                        self.train_loader, desc=f"Epoch {epoch + 1}")

                                    for batch_idx, batch in enumerate(
                                            progress_bar):
                                        images = batch['image'].to(self.device)

# Get targets for specified properties
                                        targets = []
                                        for prop in self.property_names:
                                            if prop in batch['targets']:
                                                prop_values = []
                                                for i in range(
              len(batch['targets'][prop])):
             prop_values.append(
              batch['targets'][prop][i])
             targets.append(torch.tensor(
              prop_values, dtype=torch.float32))

             if not targets:
              continue

             targets = torch.stack(
              targets, dim=1).to(self.device)

# Mixed precision forward pass
             with autocast():
              predictions = self.model(
               images)
              loss = self.criterion(
               predictions, targets)

# Scale loss for gradient accumulation
              loss = loss / \
               self.config['gradient_accumulation_steps']

# Backward pass with scaling
              self.scaler.scale(
               loss).backward()

# Update weights every N accumulation steps
              if (
                batch_idx + 1) % self.config['gradient_accumulation_steps'] == 0:
               self.scaler.step(
                self.optimizer)
               self.scaler.update()
               self.optimizer.zero_grad()

               total_loss += loss.item() * \
                self.config['gradient_accumulation_steps']
               num_batches += 1

# Update progress bar
               progress_bar.set_postfix(
                {'Loss': f"{total_loss / num_batches:.4f}"})

               avg_loss = total_loss / num_batches if num_batches > 0 else 0

               return {
                'train_loss': avg_loss}

              def validate(
                self, epoch: int) -> Dict[str, float]:
               """Validate model"""
               self.model.eval()

               total_loss = 0
               num_batches = 0
               all_predictions = []
               all_targets = []

               with torch.no_grad():
                for batch in tqdm(
                  self.val_loader, desc="Validation"):
                 images = batch['image'].to(
                  self.device)

# Get targets
                 targets = []
                 for prop in self.property_names:
                  if prop in batch['targets']:
                   prop_values = []
                   for i in range(
                     len(batch['targets'][prop])):
                    prop_values.append(
                     batch['targets'][prop][i])
                    targets.append(torch.tensor(
                     prop_values, dtype=torch.float32))

                    if not targets:
                     continue

                    targets = torch.stack(
                     targets, dim=1).to(self.device)

# Forward pass
                    predictions = self.model(
                     images)
                    loss = self.criterion(
                     predictions, targets)

                    total_loss += loss.item()
                    num_batches += 1

# Store for metrics
                    all_predictions.append(
                     predictions.cpu())
                    all_targets.append(
                     targets.cpu())

                    avg_loss = total_loss / num_batches if num_batches > 0 else 0

# Calculate additional metrics
                    if all_predictions:
                     all_predictions = torch.cat(
                      all_predictions, dim=0)
                     all_targets = torch.cat(
                      all_targets, dim=0)

# R² scores for each property
                     r2_scores = []
                     for i, prop in enumerate(
                       self.property_names):
                      pred = all_predictions[:, i]
                      target = all_targets[:, i]

# Filter out NaN values
                      mask = ~torch.isnan(
                       target)
                      if mask.sum() > 0:
                       pred_clean = pred[mask]
                       target_clean = target[mask]

                       ss_res = (
                        (target_clean - pred_clean) ** 2).sum()
                       ss_tot = (
                        (target_clean - target_clean.mean()) ** 2).sum()
                       r2 = 1 - \
                        (ss_res / ss_tot)
                       r2_scores.append(
                        r2.item())
                      else:
                       r2_scores.append(
                        0.0)
                      else:
                       r2_scores = [
                        0.0] * len(self.property_names)

                       metrics = {
                        'val_loss': avg_loss,
                        'val_r2_avg': np.mean(r2_scores)
                       }

# Add individual R² scores
                       for i, prop in enumerate(
                         self.property_names):
                        metrics[f'val_r2_{prop}'] = r2_scores[i]

                        return metrics

                       def train(
                         self, num_epochs: int) -> Dict[str, Any]:
                        """Main training loop"""
                        print(
                         f" Starting training for {num_epochs} epochs...")
                        print(
                         f"🎮 Device: {self.device}")
                        print(
                         f" Properties: {self.property_names}")

                        training_history = {
                         'train_loss': [],
                         'val_loss': [],
                         'val_r2': []
                        }

                        start_time = time.time()

                        for epoch in range(
                          num_epochs):
                         print(
                          f"\n Epoch {epoch + 1}/{num_epochs}")

# Training
                         train_metrics = self.train_epoch(
                          epoch)

# Validation
                         val_metrics = self.validate(
                          epoch)

# Learning rate scheduling
                         self.scheduler.step(
                          val_metrics['val_loss'])

# Update history
                         training_history['train_loss'].append(
                          train_metrics['train_loss'])
                         training_history['val_loss'].append(
                          val_metrics['val_loss'])
                         training_history['val_r2'].append(
                          val_metrics['val_r2_avg'])

# Logging
                         self.writer.add_scalar(
                          'Train/Loss', train_metrics['train_loss'], epoch)
                         self.writer.add_scalar(
                          'Val/Loss', val_metrics['val_loss'], epoch)
                         self.writer.add_scalar(
                          'Val/R2', val_metrics['val_r2_avg'], epoch)

# Print metrics
                         print(
                          f" 🔸 Train Loss: {train_metrics['train_loss']:.4f}")
                         print(
                          f" 🔸 Val Loss: {val_metrics['val_loss']:.4f}")
                         print(
                          f" 🔸 Val R²: {val_metrics['val_r2_avg']:.4f}")

# Early stopping check
                         if val_metrics[
                           'val_loss'] < self.best_val_loss:
                          self.best_val_loss = val_metrics[
                           'val_loss']
                          self.patience_counter = 0

# Save best model
                          torch.save({
                           'epoch': epoch,
                           'model_state_dict': self.model.state_dict(),
                           'optimizer_state_dict': self.optimizer.state_dict(),
                           'val_loss': val_metrics['val_loss'],
                           'config': self.config
                          }, f"{self.config['model_dir']}/best_model.pth")

                          print(
                           f" New best model saved! Val Loss: {self.best_val_loss:.4f}")

                         else:
                          self.patience_counter += 1
                          print(
                           f" ⏳ Patience: {self.patience_counter}/{self.config['early_stopping_patience']}")

                          if self.patience_counter >= self.config[
                            'early_stopping_patience']:
                           print(
                            f" 🛑 Early stopping triggered!")
                           break

                          total_time = (
                           time.time() - start_time) / 60  # minutes

                          print(
                           f"\n Training completed in {total_time:.1f} minutes")
                          print(
                           f" Best validation loss: {self.best_val_loss:.4f}")

                          self.writer.close()

                          return {
                           'best_val_loss': self.best_val_loss,
                           'total_epochs': epoch + 1,
                           'training_time_minutes': total_time,
                           'history': training_history
                          }

                         def main():
                          """Main function"""
                          parser = argparse.ArgumentParser(
                           description="Fine-tune vision models on molecular data")
                          parser.add_argument(
                           "--model",
                           type=str,
                           default="resnet50",
                           choices=[
                            "resnet50",
                            "resnet101",
                            "efficientnet_b0",
                            "efficientnet_b3",
                            "vit_b_16"],
                           help="Model architecture")
                          parser.add_argument(
                           "--property",
                           type=str,
                           default="logp",
                           choices=[
                            "molecular_weight",
                            "logp",
                            "hba",
                            "hbd",
                            "tpsa"],
                           help="Property to predict")
                          parser.add_argument(
                           "--epochs", type=int, default=20, help="Number of epochs")
                          parser.add_argument(
                           "--batch_size", type=int, default=32, help="Batch size")
                          parser.add_argument(
                           "--lr", type=float, default=1e-4, help="Learning rate")
                          parser.add_argument(
                           "--dataset_dir", type=str, default="drug_dataset", help="Dataset directory")

                          args = parser.parse_args()

                          print(
                           f"""
                           Molecular Vision Fine-Tuning
                           ════════════════════════════════════════
                           🤖 Model: {args.model}
                           Property: {args.property}
                           ⏱ Epochs: {args.epochs}
                           📦 Batch size: {args.batch_size}
                           Learning rate: {args.lr}
                           📁 Dataset: {args.dataset_dir}
                           ════════════════════════════════════════
                           """)

# Setup device
                          device = torch.device(
                           "cuda" if torch.cuda.is_available() else "cpu")
                          print(
                           f"🎮 Using device: {device}")

                          if device.type == "cuda":
                           print(
                            f" GPU: {torch.cuda.get_device_name()}")
                           print(
                            f" Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Prepare datasets
                           print(
                            "\n Loading datasets...")

                           config = FineTuningConfig(
                            dataset_dir=args.dataset_dir,
                            output_dir=f"{args.dataset_dir}/fine_tuning_ready",
                            prepare_vision=True,
                            image_size=224,
                            batch_size=args.batch_size,
                            property_targets=[args.property]
                           )

                           preparer = FineTuningDataPreparer(
                            config)
                           datasets = preparer.prepare_vision_datasets()

                           if not datasets:
                            print(
                             " Failed to load datasets!")
                            return 1

                           train_loader = datasets[
                            'train_loader']
                           val_loader = datasets[
                            'val_loader']

                           print(
                            f" Datasets loaded:")
                           print(
                            f" Train: {len(datasets['train_dataset'])} samples")
                           print(
                            f" Val: {len(datasets['val_dataset'])} samples")

# Create model
                           print(
                            f"\n🤖 Creating {args.model} model...")

                           model = MolecularPropertyPredictor(
                            model_name=args.model,
                            num_properties=1,  # Single property prediction
                            pretrained=True,
                            dropout=0.3
                           )

                           print(
                            f" Model created with {
                             sum(
            p.numel() for p in model.parameters()):,
                             } parameters")

# Training configuration
                           training_config = {
                            'learning_rate': args.lr,
                            'weight_decay': 1e-4,
                            'gradient_accumulation_steps': 2,  # Effective batch size = batch_size * 2
                            'early_stopping_patience': 5,
                            'model_dir': f"models/{args.model}_{args.property}",
                            'log_dir': f"logs/{args.model}_{args.property}"
                           }

# Create output directories
                           os.makedirs(
                            training_config['model_dir'], exist_ok=True)
                           os.makedirs(
                            training_config['log_dir'], exist_ok=True)

# Create trainer
                           trainer = MolecularVisionTrainer(
                            model=model,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            property_names=[args.property],
                            device=device,
                            config=training_config
                           )

# Train model
                           results = trainer.train(
                            args.epochs)

# Save results
                           results_file = f"{
                            training_config['model_dir']}/training_results.json"
                           with open(results_file, 'w') as f:
                            json.dump(
                             results, f, indent=2)

                            print(
                             f"\n Results saved to: {results_file}")
                            print(
                             f" Best model saved to: {training_config['model_dir']}/best_model.pth")

                            print(
                             f"""
                             Training Complete!
                             ════════════════════════════════════════
                             Best validation loss: {results['best_val_loss']:.4f}
                             ⏱ Total time: {results['training_time_minutes']:.1f} minutes
                             Epochs completed: {results['total_epochs']}

                             To use your trained model:
                              model = MolecularPropertyPredictor('{args.model}', 1)
                              checkpoint = torch.load('{training_config['model_dir']}/best_model.pth')
                              model.load_state_dict(checkpoint['model_state_dict'])
                              ════════════════════════════════════════
                              """)

                            if __name__ == "__main__":
                             main()
