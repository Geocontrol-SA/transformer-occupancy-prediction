#!/usr/bin/env python
"""
Inference script for Bus Occupancy Transformer model.
"""

import os
import argparse
import logging
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from occupancy_dataset import BusOccupancyDataset
from model import (
    BusOccupancyTrainer,
    collate_fn_transformer,
    create_model,
    load_config
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("inference.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference with Bus Occupancy Transformer model")
    
    # Data arguments
    parser.add_argument("--data", type=str, required=False, default="data", help="Path to test dataset")
    
    # Model arguments
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to model configuration")
    
    # Output arguments
    parser.add_argument("--output", type=str, default="predictions.csv", help="Output file path")
    parser.add_argument("--include-probabilities", action="store_true", help="Include class probabilities in output")
    
    # Other arguments
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    
    return parser.parse_args()


def load_model_for_inference(checkpoint_path, config):
    """
    Load model from checkpoint for inference.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config: The model configuration
        device: Device to load model on
        
    Returns:
        Loaded model
    """
   
    # Create model
    model = create_model(config)
    device = config['device']
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from checkpoint: {checkpoint_path}")
    logger.info(f"Model was trained for {checkpoint['epoch']} epochs")
    if 'metrics' in checkpoint:
        for metric_name, metric_value in checkpoint['metrics'].items():
            logger.info(f"  {metric_name}: {metric_value}")
    
    return model


def run_inference(model, data_loader, device):
    """
    Run inference with the model on the provided data.
    
    Args:
        model: Trained model
        data_loader: DataLoader for test data
        device: Device to run inference on
        
    Returns:
        Dictionary with predictions and metadata
    """
    # Create trainer (just for using predict method)
    trainer = BusOccupancyTrainer(
        model=model,
        train_loader=None,
        val_loader=None,
        optimizer=None,
        device=device
    )
    
    # Generate predictions
    result = trainer.predict(data_loader)
    
    return result


def save_predictions(predictions, output_file, test_dataset=None, include_probabilities=False):
    """
    Save sequence-to-sequence predictions to a CSV file.
    
    Args:
        predictions: Dictionary with prediction data
        output_file: Output file path
        test_dataset: Test dataset (for metadata)
        include_probabilities: Whether to include class probabilities
    """
    # Initialize lists for DataFrame
    trip_ids = []
    positions = []
    bus_stop_ids = []
    predicted_classes = []
    prob_class_0 = []
    prob_class_1 = []
    prob_class_2 = []
    
    # Process predictions
    for i, (trip_id, trip_positions, trip_predictions, trip_probabilities) in enumerate(zip(
        predictions['trip_ids'],
        predictions['positions'],
        predictions['predictions'],
        predictions['probabilities']
    )):
        # Get actual trip ID if available
        if test_dataset and hasattr(test_dataset, 'trip_ids'):
            trip_id = test_dataset.trip_ids[i]
        
        # Get bus stop IDs if available
        bus_stop_ids_for_trip = None
        if test_dataset and hasattr(test_dataset, 'get_bus_stop_ids'):
            bus_stop_ids_for_trip = test_dataset.get_bus_stop_ids(i)
        
        # Process each prediction in the trip
        for j, (position, prediction, probability) in enumerate(zip(
            trip_positions, trip_predictions, trip_probabilities
        )):
            trip_ids.append(trip_id)
            positions.append(position)
            
            # Add bus stop ID if available
            if bus_stop_ids_for_trip:
                bus_stop_ids.append(bus_stop_ids_for_trip[j])
            else:
                bus_stop_ids.append(None)
            
            predicted_classes.append(prediction)
            
            # Add probabilities if requested
            if include_probabilities:
                prob_class_0.append(probability[0])
                prob_class_1.append(probability[1])
                prob_class_2.append(probability[2])
    
    # Create DataFrame
    data = {
        'trip_id': trip_ids,
        'position': positions,
        'bus_stop_id': bus_stop_ids,
        'predicted_class': predicted_classes
    }
    
    # Add probabilities if requested
    if include_probabilities:
        data['prob_empty'] = prob_class_0
        data['prob_half_full'] = prob_class_1
        data['prob_full'] = prob_class_2
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    logger.info(f"Predictions saved to {output_file}")

def main():
    """Main function."""
    args = parse_args()
    
    # Load the dataset
    config = load_config(args.config)
    test_dataset_path = os.path.join(config['data_path'], 'test')
    test_dataset = BusOccupancyDataset(test_dataset_path)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn_transformer,
        num_workers=config['num_workers'],
        pin_memory=True if config['device'] == "cuda" else False
    )

    
    # Log dataset size
    logger.info(f"Test dataset size: {len(test_dataset)}")
    
    # Load model
    model = load_model_for_inference(args.checkpoint, config)
    
    # Run inference
    result = run_inference(model, test_loader, config['device'])
    
    # Extract predictions and probabilities
    predictions = result['predictions']  # This is now a list of lists
    probabilities = result['probabilities']  # This is now a list of lists
    
    # Calculate overall metrics - flatten the nested predictions first
    all_predictions_flat = [pred for trip_preds in predictions for pred in trip_preds]
    class_distribution = np.bincount(all_predictions_flat, minlength=3)
    logger.info(f"Prediction class distribution: {class_distribution}")
    logger.info(f"Total predictions: {len(all_predictions_flat)}")
    
    # Save predictions with the updated structure
    save_predictions(
        predictions=result,  # Pass the entire result dictionary
        output_file=args.output,
        test_dataset=test_dataset,
        include_probabilities=args.include_probabilities
    )
    
    logger.info("Inference completed successfully")


if __name__ == "__main__":
    main()