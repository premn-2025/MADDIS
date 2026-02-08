"""
Graph Neural Network models for molecular property prediction

Implements various GNN architectures:
- Graph Convolutional Networks (GCN)
- Graph Attention Networks (GAT)
- Message Passing Neural Networks (MPNN)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    import torch_geometric
    from torch_geometric.nn import GCNConv, GATConv, GINConv, global_mean_pool, global_max_pool
    from torch_geometric.data import Data, Batch
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    logger.warning("torch-geometric not available. GNN models will use basic implementations.")
    TORCH_GEOMETRIC_AVAILABLE = False


class MolecularGraphDataset(Dataset):
    """Dataset for molecular graphs"""

    def __init__(self, graphs: List[Dict], targets: List[float], task_type: str = 'regression'):
        self.task_type = task_type
        
        # Filter out None graphs
        valid_pairs = [(g, t) for g, t in zip(graphs, targets) if g is not None]
        self.graphs = [pair[0] for pair in valid_pairs]
        self.targets = [pair[1] for pair in valid_pairs]
        
        logger.info(f"Created dataset with {len(self.graphs)} valid molecular graphs")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        target = self.targets[idx]
        
        if TORCH_GEOMETRIC_AVAILABLE:
            data = Data(
                x=torch.FloatTensor(graph['atom_features']),
                edge_index=torch.LongTensor(graph['edge_indices']),
                edge_attr=torch.FloatTensor(graph['edge_features']) if graph['edge_features'].size > 0 else None,
                y=torch.FloatTensor([target]) if self.task_type == 'regression' else torch.LongTensor([target])
            )
            return data
        else:
            return {
                'atom_features': torch.FloatTensor(graph['atom_features']),
                'edge_indices': torch.LongTensor(graph['edge_indices']),
                'edge_features': torch.FloatTensor(graph['edge_features']) if graph['edge_features'].size > 0 else None,
                'target': torch.FloatTensor([target]) if self.task_type == 'regression' else torch.LongTensor([target]),
                'num_atoms': graph['num_atoms']
            }


class BasicGCN(nn.Module):
    """Basic Graph Convolutional Network implementation"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.convs.append(self._create_conv_layer(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(self._create_conv_layer(hidden_dim, hidden_dim))
        
        if num_layers > 1:
            self.convs.append(self._create_conv_layer(hidden_dim, hidden_dim))
        
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def _create_conv_layer(self, in_dim: int, out_dim: int):
        if TORCH_GEOMETRIC_AVAILABLE:
            return GCNConv(in_dim, out_dim)
        else:
            return nn.Linear(in_dim, out_dim)

    def forward(self, data):
        if TORCH_GEOMETRIC_AVAILABLE and hasattr(data, 'x'):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                if i < len(self.convs) - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
            
            x = global_mean_pool(x, batch)
        else:
            x = data['atom_features']
            
            for conv in self.convs:
                x = F.relu(conv(x))
                x = F.dropout(x, p=self.dropout, training=self.training)
            
            x = torch.mean(x, dim=0, keepdim=True)
        
        out = self.output(x)
        return out


class GraphAttentionNetwork(nn.Module):
    """Graph Attention Network for molecular property prediction"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 3, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_heads = num_heads
        
        if TORCH_GEOMETRIC_AVAILABLE:
            self.convs = nn.ModuleList()
            self.convs.append(GATConv(input_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout))
            
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout))
            
            if num_layers > 1:
                self.convs.append(GATConv(hidden_dim, hidden_dim, heads=1, dropout=dropout))
        else:
            self.convs = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
            for _ in range(num_layers - 1):
                self.convs.append(nn.Linear(hidden_dim, hidden_dim))
        
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, data):
        if TORCH_GEOMETRIC_AVAILABLE and hasattr(data, 'x'):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                if i < len(self.convs) - 1:
                    x = F.elu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
            
            x = global_mean_pool(x, batch)
        else:
            x = data['atom_features']
            
            for i, conv in enumerate(self.convs):
                x = conv(x)
                if i < len(self.convs) - 1:
                    x = F.elu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
            
            x = torch.mean(x, dim=0, keepdim=True)
        
        out = self.output(x)
        return out


class MessagePassingGNN(nn.Module):
    """Message Passing Graph Neural Network"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.message_layers = nn.ModuleList()
        self.update_layers = nn.ModuleList()
        
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            
            self.message_layers.append(nn.Sequential(
                nn.Linear(in_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ))
            
            self.update_layers.append(nn.Sequential(
                nn.Linear(hidden_dim + in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
        
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, data):
        if TORCH_GEOMETRIC_AVAILABLE and hasattr(data, 'x'):
            x, edge_index, batch = data.x, data.edge_index, data.batch
        else:
            x = data['atom_features']
            edge_index = data['edge_indices']
            batch = torch.zeros(x.size(0), dtype=torch.long)
        
        h = x
        for message_layer, update_layer in zip(self.message_layers, self.update_layers):
            if edge_index.size(1) > 0:
                row, col = edge_index
                messages = message_layer(torch.cat([h[row], h[col]], dim=1))
                
                aggregated = torch.zeros_like(h[:, :messages.size(1)])
                aggregated.index_add_(0, col, messages)
                
                h = update_layer(torch.cat([aggregated, h], dim=1))
            else:
                h = update_layer(torch.cat([h, h], dim=1))
        
        if TORCH_GEOMETRIC_AVAILABLE:
            graph_embedding = global_mean_pool(h, batch)
        else:
            graph_embedding = torch.mean(h, dim=0, keepdim=True)
        
        out = self.readout(graph_embedding)
        return out


class GNNPredictor:
    """High-level interface for GNN-based molecular property prediction"""

    def __init__(self, model_type: str = 'gcn', input_dim: int = 7,
                 hidden_dim: int = 64, output_dim: int = 1, **kwargs):
        
        self.model_type = model_type
        self.task_type = kwargs.get('task_type', 'regression')
        
        if model_type == 'gcn':
            self.model = BasicGCN(input_dim, hidden_dim, output_dim, **kwargs)
        elif model_type == 'gat':
            self.model = GraphAttentionNetwork(input_dim, hidden_dim, output_dim, **kwargs)
        elif model_type == 'mpnn':
            self.model = MessagePassingGNN(input_dim, hidden_dim, output_dim, **kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        if self.task_type == 'regression':
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        self.is_trained = False

    def train(self, graphs: List[Dict], targets: List[float],
              val_graphs: Optional[List[Dict]] = None,
              val_targets: Optional[List[float]] = None,
              epochs: int = 100, batch_size: int = 32) -> Dict:
        """Train the GNN model"""
        
        logger.info(f"Training {self.model_type.upper()} model...")
        
        train_dataset = MolecularGraphDataset(graphs, targets, self.task_type)
        
        if TORCH_GEOMETRIC_AVAILABLE:
            from torch_geometric.loader import DataLoader
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        else:
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        train_losses = []
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for batch in train_loader:
                if TORCH_GEOMETRIC_AVAILABLE and hasattr(batch, 'x'):
                    batch = batch.to(self.device)
                    targets_batch = batch.y
                else:
                    for key in batch:
                        if isinstance(batch[key], torch.Tensor):
                            batch[key] = batch[key].to(self.device)
                    targets_batch = batch['target']
                
                self.optimizer.zero_grad()
                outputs = self.model(batch)
                
                if self.task_type == 'regression':
                    outputs = outputs.squeeze()
                    targets_batch = targets_batch.squeeze()
                
                loss = self.criterion(outputs, targets_batch)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_loss:.4f}")
        
        self.is_trained = True
        
        results = {
            'train_losses': train_losses,
            'final_train_loss': train_losses[-1]
        }
        
        logger.info(f"GNN training completed. Final train loss: {train_losses[-1]:.4f}")
        return results

    def predict(self, graphs: List[Dict]) -> np.ndarray:
        """Make predictions on new graphs"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        dummy_targets = [0.0] * len(graphs)
        dataset = MolecularGraphDataset(graphs, dummy_targets, self.task_type)
        
        if TORCH_GEOMETRIC_AVAILABLE:
            from torch_geometric.loader import DataLoader
            loader = DataLoader(dataset, batch_size=32, shuffle=False)
        else:
            loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
        
        predictions = []
        self.model.eval()
        
        with torch.no_grad():
            for batch in loader:
                if TORCH_GEOMETRIC_AVAILABLE and hasattr(batch, 'x'):
                    batch = batch.to(self.device)
                else:
                    for key in batch:
                        if isinstance(batch[key], torch.Tensor):
                            batch[key] = batch[key].to(self.device)
                
                outputs = self.model(batch)
                
                if self.task_type == 'regression':
                    preds = outputs.squeeze().cpu().numpy()
                else:
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                
                predictions.extend(preds)
        
        return np.array(predictions)


if __name__ == "__main__":
    # Test with sample data
    sample_graphs = [
        {
            'atom_features': np.random.rand(5, 7),
            'edge_indices': np.array([[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]]),
            'edge_features': np.random.rand(8, 3),
            'num_atoms': 5
        }
        for _ in range(100)
    ]
    
    sample_targets = np.random.rand(100)
    
    gnn = GNNPredictor('gcn', input_dim=7, hidden_dim=32)
    results = gnn.train(sample_graphs, sample_targets, epochs=10)
    
    predictions = gnn.predict(sample_graphs[:10])
    print(f"Sample predictions: {predictions[:5]}")
    print(f"Training completed with final loss: {results['final_train_loss']:.4f}")
