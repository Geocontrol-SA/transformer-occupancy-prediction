from datetime import datetime
import json
import os
from typing import Any, Dict, List, Optional

from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import torch

class DualMetricsTracker:
    """Tracks both training and validation metrics with disk persistence."""
    def __init__(self, save_dir: str, class_names: List[str]):
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.class_names = class_names
        self.reset_all()
        
    def reset_all(self):
        """Reset all accumulators."""
        self.train = _EpochTracker(class_names=self.class_names)
        self.val = _EpochTracker(class_names=self.class_names)
    
    def update(self, 
               phase: str,  # 'train' or 'val'
               loss: float, 
               targets: torch.Tensor, 
               outputs: torch.Tensor,
               masks: Optional[torch.Tensor] = None):
        """Update metrics for a specific phase."""
        tracker = getattr(self, phase)
        tracker.update(loss, targets, outputs, masks)

    def save_epoch(self, epoch: int):
        """Persist metrics for both phases."""
        metrics = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            "train": self.train.compute_metrics(),
            "val": self.val.compute_metrics()
        }
        
        with open(os.path.join(self.save_dir, f"metrics_epoch_{epoch}.json"), "w") as f:
            json.dump(metrics, f, indent=2)

class _EpochTracker:
    """Internal class for per-phase tracking."""
    def __init__(self, class_names: List[str]):
        self.reset()
        self.class_names = class_names
    
    def reset(self):
        self.loss = 0.0
        self.targets = []
        self.predictions = []
        self.total_samples = 0
    
    def update(self, 
               loss: float, 
               targets: torch.Tensor, 
               outputs: torch.Tensor,
               masks: Optional[torch.Tensor] = None):
        """Update metrics with batch results."""
        self.loss += loss * targets.size(0)
        self.total_samples += targets.size(0)
        
        # Handle sequence data with masks
        if masks is not None:
            valid_mask = ~masks.reshape(-1)
            targets = targets.reshape(-1)[valid_mask]
            outputs = outputs.reshape(-1, outputs.shape[-1])[valid_mask]
        
        preds = outputs.argmax(dim=1)
        self.targets.extend(targets.cpu().numpy())
        self.predictions.extend(preds.cpu().numpy())

    def compute_metrics(self) -> Dict[str, Any]:
        """Compute all metrics for the accumulated data."""
        if not self.targets:
            return {}
            
        metrics = {
            "loss": self.loss / self.total_samples,
            "accuracy": accuracy_score(self.targets, self.predictions),
            "precision_macro": precision_score(self.targets, self.predictions, average="macro", zero_division=0),
            "recall_macro": recall_score(self.targets, self.predictions, average="macro", zero_division=0),
            "f1_macro": f1_score(self.targets, self.predictions, average="macro", zero_division=0),
            "confusion_matrix": confusion_matrix(self.targets, self.predictions).tolist(),
            "classification_report": classification_report(
                self.targets, self.predictions, 
                target_names=self.class_names,
                output_dict=True,
                zero_division=0
            )
        }
        return metrics

    def save_epoch(self, epoch: int):
        """Persist metrics to disk."""
        metrics = self.compute_metrics()
        if not metrics:
            return
            
        # Save as JSON with timestamp
        timestamp = datetime.now().isoformat()
        filename = f"metrics_epoch_{epoch}_{timestamp}.json"
        with open(os.path.join(self.save_dir, filename), "w") as f:
            json.dump({
                "epoch": epoch,
                "timestamp": timestamp,
                **metrics
            }, f, indent=2)