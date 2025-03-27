"""
Bus Occupancy Transformer

A transformer-based model for predicting bus occupancy levels based on sequential trip data.
"""

import os
import logging
from typing import Dict, Tuple, List, Optional, Any
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from metrics_calc import DualMetricsTracker

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bus_occupancy_transformer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    """
    Positional encoding for the transformer model.
    Adds positional information to the input embeddings.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Initialize the positional encoding.
        
        Args:
            d_model: Embedding dimension
            dropout: Dropout rate
            max_len: Maximum sequence length
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor.
        
        Args:
            x: Input tensor of shape [seq_len, batch_size, embedding_dim]
            
        Returns:
            Output tensor with positional encoding added
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class BusOccupancyTransformer(nn.Module):
    """
    Transformer model for predicting bus occupancy levels.
    Uses a transformer encoder architecture with a classification head.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        nhead: int,
        num_encoder_layers: int,
        num_classes: int = 3,
        dropout: float = 0.1,
        use_positional_encoding: bool = True,
        max_seq_len: int = 100,
        use_cls_token: bool = True
    ):
        """
        Initialize the transformer model.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layers
            nhead: Number of attention heads
            num_encoder_layers: Number of transformer encoder layers
            num_classes: Number of output classes
            dropout: Dropout rate
            use_positional_encoding: Whether to use positional encoding
            max_seq_len: Maximum sequence length
            use_cls_token: Whether to use a CLS token for classification
        """
        super().__init__()
        self.use_positional_encoding = use_positional_encoding
        self.use_cls_token = use_cls_token
        self.hidden_dim = hidden_dim
        
        # Input embedding layer
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        if use_positional_encoding:
            self.pos_encoder = PositionalEncoding(hidden_dim, dropout, max_seq_len)
        
        # CLS token for classification (learnable parameter)
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_encoder_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize the model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape [seq_len, batch_size, feature_dim]
            mask: Mask tensor for padding, shape [batch_size, seq_len]
            
        Returns:
            Output tensor with logits for each position, shape [batch_size, seq_len, num_classes]
        """
        # [seq_len, batch_size, feature_dim] -> [seq_len, batch_size, hidden_dim]
        x = self.embedding(x)
        
        if self.use_positional_encoding:
            x = self.pos_encoder(x)
        
        # Apply transformer encoder
        # [seq_len, batch_size, hidden_dim] -> [seq_len, batch_size, hidden_dim]
        if mask is not None:
            x = self.transformer_encoder(x, src_key_padding_mask=mask)
        else:
            x = self.transformer_encoder(x)
        
        # Reshape for classification
        # [seq_len, batch_size, hidden_dim] -> [batch_size, seq_len, hidden_dim]
        x = x.transpose(0, 1)
        
        # Apply classifier to each position independently
        batch_size, seq_len, _ = x.shape
        
        # Reshape to apply classifier efficiently
        x_reshaped = x.reshape(-1, self.hidden_dim)  # [batch_size*seq_len, hidden_dim]
        x_classified = self.classifier(x_reshaped)  # [batch_size*seq_len, num_classes]
        
        # Reshape back to sequence format
        x_output = x_classified.reshape(batch_size, seq_len, -1)  # [batch_size, seq_len, num_classes]
        
        return x_output


class BusOccupancyTrainer:
    """
    Trainer class for the Bus Occupancy Transformer model.
    Handles training, evaluation, checkpointing, and inference.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "checkpoints",
        class_weights: Optional[torch.Tensor] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            optimizer: Optimizer for training
            scheduler: Learning rate scheduler (optional)
            device: Device to train on ('cuda' or 'cpu')
            checkpoint_dir: Directory to save checkpoints
            class_weights: Weights for imbalanced classes (optional)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Set up criterion with class weights if provided
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.metrics_tracker = DualMetricsTracker(
            save_dir=os.path.join(checkpoint_dir, "metrics"),
            class_names=['Empty', 'Half Full', 'Full']
        )
        
        # Move model to device
        self.model.to(device)
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train the model for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        self.metrics_tracker.reset_all()
        for batch_idx, (features, masks, targets) in enumerate(pbar):
            # Move data to device
            features = features.to(self.device)
            if masks is not None:
                masks = masks.to(self.device)
            targets = targets.to(self.device)
            
            # Clear gradients
            self.optimizer.zero_grad()
            
            # Forward pass - handle different shapes
            if features.dim() == 3:  # [batch_size, seq_len, features]
                # Transpose to [seq_len, batch_size, features]
                features = features.transpose(0, 1)
            
            # Forward pass
            outputs = self.model(features, masks)  # [batch_size, seq_len, num_classes]
            
            # Prepare for loss calculation
            batch_size, seq_len, num_classes = outputs.shape
            outputs_flat = outputs.reshape(-1, num_classes)  # [batch_size*seq_len, num_classes]
            targets_flat = targets.reshape(-1)  # [batch_size*seq_len]
            
            # Create mask for valid positions (not padding)
            if masks is not None:
                valid_mask = ~masks.reshape(-1)  # [batch_size*seq_len]
                outputs_flat = outputs_flat[valid_mask]
                targets_flat = targets_flat[valid_mask]
            
            # Compute loss on all valid positions
            loss = self.criterion(outputs_flat, targets_flat)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            _, predicted_flat = outputs_flat.max(1)
            total += targets_flat.size(0)
            correct += predicted_flat.eq(targets_flat).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{total_loss/(batch_idx+1):.4f}",
                'acc': f"{100.*correct/total:.2f}%"
            })
            self.metrics_tracker.update('train', loss.item(), targets_flat, outputs_flat)
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        # Store metrics
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(accuracy)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def evaluate(self, data_loader: DataLoader, phase: str = "val") -> Dict[str, Any]:
        """
        Evaluate the model on the provided data loader.
        Modified for sequence-to-sequence prediction.
        
        Args:
            data_loader: DataLoader for evaluation
            phase: Phase name for logging ('val' or 'test')
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_targets = []
        all_predictions = []
        
        # Use tqdm for progress bar
        pbar = tqdm(data_loader, desc=f"[{phase.capitalize()}]")
        
        with torch.no_grad():
            for features, masks, targets in pbar:
                # Move data to device
                features = features.to(self.device)
                if masks is not None:
                    masks = masks.to(self.device)
                targets = targets.to(self.device)
                
                # Handle different shapes depending on dataset structure
                if features.dim() == 3:  # [batch_size, seq_len, features]
                    # Transpose to [seq_len, batch_size, features]
                    features = features.transpose(0, 1)
                
                # Forward pass
                outputs = self.model(features, masks)  # [batch_size, seq_len, num_classes]
                
                # Prepare for loss calculation
                batch_size, seq_len, num_classes = outputs.shape
                outputs_flat = outputs.reshape(-1, num_classes)  # [batch_size*seq_len, num_classes]
                targets_flat = targets.reshape(-1)  # [batch_size*seq_len]
                
                # Create mask for valid positions (not padding)
                if masks is not None:
                    valid_mask = ~masks.reshape(-1)  # [batch_size*seq_len]
                    outputs_flat = outputs_flat[valid_mask]
                    targets_flat = targets_flat[valid_mask]
                
                # Compute loss on all valid positions
                loss = self.criterion(outputs_flat, targets_flat)
                total_loss += loss.item()
                
                # Get predictions
                _, predicted_flat = outputs_flat.max(1)
                
                # Store targets and predictions for metrics
                all_targets.extend(targets_flat.cpu().numpy())
                all_predictions.extend(predicted_flat.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{total_loss/(pbar.n+1):.4f}"
                })
                self.metrics_tracker.update(phase, loss.item(), targets_flat, outputs_flat)
        
        # Calculate metrics
        avg_loss = total_loss / len(data_loader)
        accuracy = 100. * accuracy_score(all_targets, all_predictions)
        f1 = 100. * f1_score(all_targets, all_predictions, average='weighted')
        
        # Store validation metrics
        if phase == "val":
            self.val_losses.append(avg_loss)
            self.val_accuracies.append(accuracy)
        
        # Generate classification report
        report = classification_report(
            all_targets, 
            all_predictions, 
            target_names=['Empty', 'Half Full', 'Full'], 
            output_dict=True,
            zero_division=0
        )
        conf_matrix = confusion_matrix(all_targets, all_predictions)
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1_score': f1,
            'classification_report': report,
            'confusion_matrix': conf_matrix
        }
        
        return metrics
    
    def train(self, num_epochs: int, early_stopping_patience: int = 10) -> Dict[str, List[float]]:
        """
        Train the model for the specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Number of epochs to wait before early stopping
            
        Returns:
            Dictionary with training history
        """
        logger.info(f"Starting training for {num_epochs} epochs on {self.device}")
        
        # Initialize early stopping variables
        best_val_loss = float('inf')
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Train for one epoch
            train_metrics = self.train_epoch(epoch)
            
            # Evaluate on validation set
            val_metrics = self.evaluate(self.val_loader, phase="val")
            
            # Update learning rate if scheduler is provided
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Log metrics
            logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                        f"Train Loss: {train_metrics['loss']:.4f}, "
                        f"Train Acc: {train_metrics['accuracy']:.2f}%, "
                        f"Val Loss: {val_metrics['loss']:.4f}, "
                        f"Val Acc: {val_metrics['accuracy']:.2f}%, "
                        f"Val F1: {val_metrics['f1_score']:.2f}%")
            
            # Print confusion matrix
            logger.info(f"Confusion Matrix:\n{val_metrics['confusion_matrix']}")
            
            # Check if this is the best model
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                logger.info(f"New best accuracy: {best_val_acc:.2f}%")
                
                # Save best model
                self.save_checkpoint("best_model_acc.pt", epoch, val_metrics)
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Check for early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint("last_model.pt", epoch, val_metrics)
            self.metrics_tracker.save_epoch(epoch)
        
        logger.info("Training completed")
        
        # Return training history
        return {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'train_acc': self.train_accuracies,
            'val_acc': self.val_accuracies
        }
    
    def save_checkpoint(self, filename: str, epoch: int, metrics: Dict[str, Any]) -> None:
        """
        Save a checkpoint of the model.
        
        Args:
            filename: Name of the checkpoint file
            epoch: Current epoch number
            metrics: Dictionary with evaluation metrics
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        
        # Create checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': {k: v for k, v in metrics.items() if isinstance(v, (int, float, str))},
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
        
        # Add scheduler state if available
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        # Save metrics to a JSON file
        metrics_path = os.path.join(self.checkpoint_dir, f"{os.path.splitext(filename)[0]}_metrics.json")
        # Create a copy of metrics for saving
        metrics_to_save = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float, str)):
                metrics_to_save[k] = v
            elif k == 'classification_report':
                # Convert classification report to JSON-serializable format
                metrics_to_save[k] = v
            elif k == 'confusion_matrix':
                # Convert numpy array to list
                metrics_to_save[k] = v.tolist()

        # Add timestamp and epoch information
        metrics_to_save["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        metrics_to_save["epoch"] = epoch

        # Save to JSON file
        with open(metrics_path, 'w') as f:
            json.dump(metrics_to_save, f, indent=4)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        load_optimizer: bool = True,
        load_scheduler: bool = True
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            load_optimizer: Whether to load optimizer state
            load_scheduler: Whether to load scheduler state
            
        Returns:
            Dictionary with checkpoint information
        """
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if requested
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if requested and available
        if load_scheduler and self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training history
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
        if 'val_losses' in checkpoint:
            self.val_losses = checkpoint['val_losses']
        if 'train_accuracies' in checkpoint:
            self.train_accuracies = checkpoint['train_accuracies']
        if 'val_accuracies' in checkpoint:
            self.val_accuracies = checkpoint['val_accuracies']
        
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        
        return checkpoint
    
    def predict(self, data_loader: DataLoader) -> Dict[str, Any]:
        """
        Generate predictions for the provided data loader.
        Modified for sequence-to-sequence prediction.
        
        Args:
            data_loader: DataLoader for inference
            
        Returns:
            Dictionary with predictions and probabilities for each position
        """
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        all_trip_ids = []
        all_positions = []
        
        # Use tqdm for progress bar
        pbar = tqdm(data_loader, desc="Generating predictions")
        
        with torch.no_grad():
            for batch_idx, (features, masks, _) in enumerate(pbar):
                # Move data to device
                features = features.to(self.device)
                if masks is not None:
                    masks = masks.to(self.device)
                
                # Handle different shapes depending on dataset structure
                if features.dim() == 3:  # [batch_size, seq_len, features]
                    # Transpose to [seq_len, batch_size, features]
                    features = features.transpose(0, 1)
                
                # Forward pass
                outputs = self.model(features, masks)  # [batch_size, seq_len, num_classes]
                
                # Get predictions and probabilities
                batch_size, seq_len, num_classes = outputs.shape
                
                # Process each item in the batch
                for i in range(batch_size):
                    trip_predictions = []
                    trip_probabilities = []
                    positions = []
                    
                    # Process each position in the sequence
                    for j in range(seq_len):
                        # Skip padded positions
                        if masks is not None and masks[i, j]:
                            continue
                        
                        # Get prediction and probability for this position
                        logits = outputs[i, j]
                        probs = F.softmax(logits, dim=0)
                        _, pred = logits.max(0)
                        
                        # Store values
                        trip_predictions.append(pred.item())
                        trip_probabilities.append(probs.cpu().numpy())
                        positions.append(j)
                    
                    # Store trip data
                    all_predictions.append(trip_predictions)
                    all_probabilities.append(trip_probabilities)
                    all_trip_ids.append(batch_idx * batch_size + i)  # Create a simple ID if not available
                    all_positions.append(positions)
        
        return {
            'predictions': all_predictions,       # List of lists, each containing predictions for one trip
            'probabilities': all_probabilities,   # List of lists, each containing probability vectors for one trip
            'trip_ids': all_trip_ids,             # List of trip IDs 
            'positions': all_positions            # List of lists, each containing position indices for one trip
        }


def collate_fn_transformer(batch):
    """
    Collate function for transformer model.
    Takes a batch of trip data and prepares it for the transformer model.
    
    Args:
        batch: List of tuples (features, targets) for each trip
        
    Returns:
        Tuple of tensors (features, masks, targets)
    """
    batch_size = len(batch)
    
    # Find the maximum sequence length in the batch
    max_seq_len = max(len(trip[0]) for trip in batch)
    
    # Initialize tensors
    feature_dim = len(batch[0][0][0])
    features = torch.zeros((batch_size, max_seq_len, feature_dim), dtype=torch.float32)
    masks = torch.ones((batch_size, max_seq_len), dtype=torch.bool)
    targets = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    
    # Fill tensors with data
    for i, (trip_features, trip_targets) in enumerate(batch):
        seq_len = len(trip_features)
        features[i, :seq_len] = torch.tensor(trip_features, dtype=torch.float32)
        targets[i, :seq_len] = torch.tensor(trip_targets, dtype=torch.long)
        masks[i, :seq_len] = False  # Set to False (not masked) for actual data
    
    return features, masks, targets


def create_model(config):
    """
    Create the transformer model based on the provided configuration.
    
    Args:
        config: Dictionary with model configuration
        
    Returns:
        Initialized model
    """
    model = BusOccupancyTransformer(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        nhead=config['nhead'],
        num_encoder_layers=config['num_encoder_layers'],
        num_classes=config['num_classes'],
        dropout=config['dropout'],
        use_positional_encoding=config['use_positional_encoding'],
        max_seq_len=config['max_seq_len'],
        use_cls_token=config['use_cls_token']
    )
    
    return model


def create_optimizer(model, config):
    """
    Create optimizer based on the provided configuration.
    
    Args:
        model: Model to optimize
        config: Dictionary with optimizer configuration
        
    Returns:
        Initialized optimizer
    """
    if config['optimizer'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
    elif config['optimizer'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config['optimizer']}")
    
    return optimizer


def create_scheduler(optimizer, config):
    """
    Create learning rate scheduler based on the provided configuration.
    
    Args:
        optimizer: Optimizer to schedule
        config: Dictionary with scheduler configuration
        
    Returns:
        Initialized scheduler
    """
    if config['scheduler'] == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config['scheduler_factor'],
            patience=config['scheduler_patience'],
            min_lr=config['min_lr'],
            verbose=True
        )
    elif config['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['T_max'],
            eta_min=config['min_lr']
        )
    elif config['scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['step_size'],
            gamma=config['scheduler_factor']
        )
    else:
        scheduler = None
    
    return scheduler


def compute_class_weights(train_loader, num_classes=3, device='cpu'):
    """
    Compute class weights based on class distribution in the training data.
    
    Args:
        train_loader: Training data loader
        num_classes: Number of classes
        device: Device to place tensor on
        
    Returns:
        Tensor with class weights
    """
    class_counts = torch.zeros(num_classes)
    
    # Count occurrences of each class
    for _, _, targets in train_loader:
        if targets.dim() == 2:  # [batch_size, seq_len]
            # Flatten targets to count all occurrences
            targets_flat = targets.view(-1)
            # Count only non-padded values (if there's a mask, but we don't have it here)
            # Just count all values for simplicity
            for c in range(num_classes):
                class_counts[c] += (targets_flat == c).sum().item()
        else:
            for c in range(num_classes):
                class_counts[c] += (targets == c).sum().item()
    
    # Compute inverse frequency
    total_samples = class_counts.sum()
    class_weights = total_samples / (class_counts * num_classes)
    
    # Handle classes with zero samples
    class_weights[class_counts == 0] = 0.0
    
    logger.info(f"Class distribution: {class_counts.tolist()}")
    logger.info(f"Class weights: {class_weights.tolist()}")
    
    return class_weights.to(device)


def save_config(config, filepath):
    """
    Save configuration to a JSON file.
    
    Args:
        config: Configuration dictionary
        filepath: Path to save the configuration
    """
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=4)
    logger.info(f"Configuration saved to {filepath}")


def load_config(filepath):
    """
    Load configuration from a JSON file.
    
    Args:
        filepath: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(filepath, 'r') as f:
        config = json.load(f)
    logger.info(f"Configuration loaded from {filepath}")
    return config