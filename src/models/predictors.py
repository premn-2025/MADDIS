"""
Machine Learning Models for Drug Discovery Prediction Tasks

Implements various ML architectures for:
- Binding affinity prediction
- ADMET properties prediction
- Toxicity prediction
- Drug-likeness scoring
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, roc_auc_score
from typing import Dict, List, Optional, Tuple, Union
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class DrugDataset(Dataset):
    """PyTorch Dataset for drug discovery data"""

    def __init__(self, features: np.ndarray, targets: np.ndarray, task_type: str = 'regression'):
        self.features = torch.FloatTensor(features)
        
        if task_type == 'regression':
            self.targets = torch.FloatTensor(targets)
        else:
            self.targets = torch.LongTensor(targets)
        
        self.task_type = task_type

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class BasePredictor(ABC):
    """Abstract base class for drug discovery predictors"""

    def __init__(self, task_type: str = 'regression'):
        self.task_type = task_type
        self.model = None
        self.is_trained = False

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict:
        """Train the model"""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate model performance"""
        predictions = self.predict(X_test)
        
        if self.task_type == 'regression':
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            return {'mse': mse, 'rmse': np.sqrt(mse), 'r2': r2}
        else:
            accuracy = accuracy_score(y_test, predictions)
            
            if len(np.unique(y_test)) == 2:
                try:
                    probs = self.predict_proba(X_test)[:, 1]
                    auc = roc_auc_score(y_test, probs)
                    return {'accuracy': accuracy, 'auc': auc}
                except Exception:
                    return {'accuracy': accuracy}
            
            return {'accuracy': accuracy}


class RandomForestPredictor(BasePredictor):
    """Random Forest predictor for drug discovery tasks"""

    def __init__(self, task_type: str = 'regression', **kwargs):
        super().__init__(task_type)
        
        if task_type == 'regression':
            self.model = RandomForestRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10),
                random_state=kwargs.get('random_state', 42),
                n_jobs=-1
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10),
                random_state=kwargs.get('random_state', 42),
                n_jobs=-1
            )

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict:
        """Train Random Forest model"""
        logger.info("Training Random Forest model...")
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        train_metrics = self.evaluate(X_train, y_train)
        results = {'train_metrics': train_metrics}
        
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            results['val_metrics'] = val_metrics
        
        logger.info(f"Random Forest training completed. Train R2/Acc: {train_metrics}")
        return results

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities (classification only)"""
        if self.task_type != 'classification':
            raise ValueError("Probabilities only available for classification tasks")
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        return self.model.feature_importances_


class MLPPredictor(BasePredictor):
    """Multi-layer Perceptron predictor using scikit-learn"""

    def __init__(self, task_type: str = 'regression', **kwargs):
        super().__init__(task_type)
        
        hidden_layer_sizes = kwargs.get('hidden_layer_sizes', (100, 50))
        
        if task_type == 'regression':
            self.model = MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                max_iter=kwargs.get('max_iter', 1000),
                alpha=kwargs.get('alpha', 0.001),
                random_state=kwargs.get('random_state', 42),
                early_stopping=True,
                validation_fraction=0.1
            )
        else:
            self.model = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                max_iter=kwargs.get('max_iter', 1000),
                alpha=kwargs.get('alpha', 0.001),
                random_state=kwargs.get('random_state', 42),
                early_stopping=True,
                validation_fraction=0.1
            )

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict:
        """Train MLP model"""
        logger.info("Training MLP model...")
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        train_metrics = self.evaluate(X_train, y_train)
        results = {'train_metrics': train_metrics}
        
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            results['val_metrics'] = val_metrics
        
        logger.info(f"MLP training completed. Train R2/Acc: {train_metrics}")
        return results

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities (classification only)"""
        if self.task_type != 'classification':
            raise ValueError("Probabilities only available for classification tasks")
        return self.model.predict_proba(X)


class DeepNeuralNetwork(nn.Module):
    """Deep Neural Network for drug discovery predictions"""

    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int,
                 task_type: str = 'regression', dropout: float = 0.2):
        super().__init__()
        
        self.task_type = task_type
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        if task_type == 'classification' and output_size > 1:
            layers.append(nn.Softmax(dim=1))
        elif task_type == 'classification' and output_size == 1:
            layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class DeepPredictor(BasePredictor):
    """Deep learning predictor using PyTorch"""

    def __init__(self, input_size: int, task_type: str = 'regression', **kwargs):
        super().__init__(task_type)
        
        self.input_size = input_size
        self.hidden_sizes = kwargs.get('hidden_sizes', [128, 64, 32])
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.batch_size = kwargs.get('batch_size', 32)
        self.epochs = kwargs.get('epochs', 100)
        self.dropout = kwargs.get('dropout', 0.2)
        
        if task_type == 'regression':
            output_size = 1
        else:
            output_size = kwargs.get('num_classes', 2)
        
        self.model = DeepNeuralNetwork(input_size, self.hidden_sizes, output_size, task_type, self.dropout)
        
        if task_type == 'regression':
            self.criterion = nn.MSELoss()
        else:
            if output_size > 2:
                self.criterion = nn.CrossEntropyLoss()
            else:
                self.criterion = nn.BCELoss()
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict:
        """Train deep neural network"""
        logger.info("Training Deep Neural Network...")
        
        train_dataset = DrugDataset(X_train, y_train, self.task_type)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        train_losses = []
        
        for epoch in range(self.epochs):
            self.model.train()
            epoch_train_loss = 0.0
            
            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_features)
                
                if self.task_type == 'regression':
                    outputs = outputs.squeeze()
                
                loss = self.criterion(outputs, batch_targets)
                loss.backward()
                self.optimizer.step()
                
                epoch_train_loss += loss.item()
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{self.epochs}, Train Loss: {avg_train_loss:.4f}")
        
        self.is_trained = True
        
        train_metrics = self.evaluate(X_train, y_train)
        results = {
            'train_metrics': train_metrics,
            'train_losses': train_losses
        }
        
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            results['val_metrics'] = val_metrics
        
        logger.info(f"Deep NN training completed. Train metrics: {train_metrics}")
        return results

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        self.model.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            
            if self.task_type == 'regression':
                predictions = outputs.squeeze().cpu().numpy()
            else:
                if outputs.shape[1] > 1:
                    predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                else:
                    predictions = (outputs.squeeze() > 0.5).cpu().numpy().astype(int)
        
        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities (classification only)"""
        if self.task_type != 'classification':
            raise ValueError("Probabilities only available for classification tasks")
        
        self.model.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            probabilities = outputs.cpu().numpy()
        
        return probabilities


class PropertyPredictor:
    """High-level interface for property prediction"""

    def __init__(self, model_type: str = 'random_forest', task_type: str = 'regression', **kwargs):
        self.model_type = model_type
        self.task_type = task_type
        
        if model_type == 'random_forest':
            self.predictor = RandomForestPredictor(task_type, **kwargs)
        elif model_type == 'mlp':
            self.predictor = MLPPredictor(task_type, **kwargs)
        elif model_type == 'deep':
            input_size = kwargs.get('input_size')
            if input_size is None:
                raise ValueError("input_size required for deep neural network")
            self.predictor = DeepPredictor(input_size, task_type, **kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict:
        """Train the predictor"""
        return self.predictor.train(X_train, y_train, X_val, y_val)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.predictor.predict(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate performance"""
        return self.predictor.evaluate(X_test, y_test)


if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    X_train = np.random.rand(1000, 50)
    y_train_reg = np.random.rand(1000)
    y_train_clf = np.random.randint(0, 2, 1000)
    
    X_test = np.random.rand(200, 50)
    y_test_reg = np.random.rand(200)
    y_test_clf = np.random.randint(0, 2, 200)
    
    # Test regression predictor
    print("Testing Regression Predictor...")
    reg_predictor = PropertyPredictor('random_forest', 'regression')
    reg_results = reg_predictor.train(X_train, y_train_reg)
    reg_test_metrics = reg_predictor.evaluate(X_test, y_test_reg)
    print(f"Regression test metrics: {reg_test_metrics}")
    
    # Test classification predictor
    print("\nTesting Classification Predictor...")
    clf_predictor = PropertyPredictor('random_forest', 'classification')
    clf_results = clf_predictor.train(X_train, y_train_clf)
    clf_test_metrics = clf_predictor.evaluate(X_test, y_test_clf)
    print(f"Classification test metrics: {clf_test_metrics}")
