#!/usr/bin/env python
"""
Training script for Bus Occupancy Transformer model.
"""

import os
import argparse
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from model import (
    BusOccupancyTrainer,
    collate_fn_transformer,
    create_model,
    create_optimizer,
    create_scheduler,
    compute_class_weights,
    save_config,
    load_config
)
from iterable_occupancy_dataset import create_dataset_from_processed_dir
import shutil

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("train.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Bus Occupancy Transformer model")
    
    # Data arguments
    parser.add_argument("--data-path", type=str, required=False, help="Path to all data needed")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for data loading")
    
    # Model arguments
    parser.add_argument("--input-dim", type=int, default=13, help="Input dimension")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num-encoder-layers", type=int, default=4, help="Number of encoder layers")
    parser.add_argument("--num-classes", type=int, default=3, help="Number of output classes")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--use-positional-encoding", action="store_true", help="Use positional encoding")
    parser.add_argument("--max-seq-len", type=int, default=130, help="Maximum sequence length")
    parser.add_argument("--use-cls-token", action="store_true", help="Use CLS token for classification")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--optimizer", type=str, default="adamw", help="Optimizer (adam, adamw)")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--scheduler", type=str, default="plateau", help="LR scheduler (plateau, cosine, step, none)")
    parser.add_argument("--scheduler-factor", type=float, default=0.5, help="Scheduler factor")
    parser.add_argument("--scheduler-patience", type=int, default=5, help="Scheduler patience")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Minimum learning rate")
    parser.add_argument("--T-max", type=int, default=10, help="T_max for cosine scheduler")
    parser.add_argument("--step-size", type=int, default=10, help="Step size for step scheduler")
    parser.add_argument("--early-stopping-patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--use-class-weights", action="store_true", help="Use class weights for loss")
    
    # Other arguments
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--config", type=str, help="Configuration file")
    parser.add_argument("--resume", type=str, help="Resume training from checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args, unknown = parser.parse_known_args()
    if unknown:
        logger.info(f"Ignoring unknown arguments: {unknown}")
    return args


def main(config_file=None):
    """Main function."""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Check if checkpoint directory exists and has files
    checkpoint_dir = Path(args.checkpoint_dir)
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        user_input = input(f"Checkpoint directory {args.checkpoint_dir} is not empty. Clear it? (Y/n): ")
        if user_input.lower() != 'n':
            logger.warning("You are about to delete all files in the checkpoint directory!")
            logger.warning("Please make a backup if you have important models saved.")
            choice = input("Press Enter to continue or 'n' to cancel: ")
            if choice.lower() == 'n':
                logger.info("Operation cancelled")
                return
            logger.info(f"Clearing checkpoint directory {args.checkpoint_dir}")
            for file_path in checkpoint_dir.iterdir():
                if file_path.is_file():
                    file_path.unlink()
                elif file_path.is_dir():
                    shutil.rmtree(file_path)
            logger.info("Directory cleared")
        else:
            logger.info("Keeping existing checkpoint files")

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Load configuration from file if provided
    config_file = config_file if config_file is not None else args.config
    if config_file:
        config = load_config(config_file)
        # Update args with config values
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
        logger.info("Loaded configuration from file")
    else:
        # Create configuration from arguments
        config = vars(args)
    if args.data_path is None:
        logger.error("Please provide a data path")
        return
    # Save configuration
    save_config(config, os.path.join(args.checkpoint_dir, "config.json"))
    
    # Load your dataset (assuming it's implemented)
    # Replace this with your actual dataset implementation
    try:
        train_dataset, _ = create_dataset_from_processed_dir(args.data_path, "train")
        val_dataset, _ = create_dataset_from_processed_dir(args.data_path, "val")
        
        dataset_max_seq_len = max(train_dataset.max_seq_length, val_dataset.max_seq_length)
        if args.max_seq_len != dataset_max_seq_len:
            logger.warning(f"Maximum sequence length is different from dataset maximum sequence length. "
                           f"Setting max_seq_len to {dataset_max_seq_len}")
            args.max_seq_len = dataset_max_seq_len

       
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn_transformer,
            num_workers=args.num_workers,
            pin_memory=True if args.device == "cuda" else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn_transformer,
            num_workers=args.num_workers,
            pin_memory=True if args.device == "cuda" else False
        )
        
    except ImportError:
        logger.error("Failed to import dataset module. Please implement your dataset class.")
        raise
    
    # Log dataset sizes
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # Create model
    model = create_model(config)
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create optimizer
    optimizer = create_optimizer(model, config)
    
    # Compute class weights if needed
    class_weights = None
    if args.use_class_weights:
        # class_weights = compute_class_weights(train_loader, args.num_classes, args.device)
        # Save class weights to file
        class_weights_file = os.path.join(args.checkpoint_dir, "class_weights.pt")
        if os.path.exists(class_weights_file):
            logger.info(f"Loading class weights from {class_weights_file}")
            class_weights = torch.load(class_weights_file)
        else:
            logger.info(f"Computing class weights and saving to {class_weights_file}")
            class_weights = compute_class_weights(train_loader, args.num_classes, args.device)
            # Disable saving class weights for now. It can be obtained from stats.json file
            # torch.save(class_weights, class_weights_file)
        logger.info(f"Class weights: {class_weights}")
    
    # Create scheduler
    scheduler = create_scheduler(optimizer, config)
    
    # Create trainer
    trainer = BusOccupancyTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        class_weights=class_weights
    )
    
    # Resume training if checkpoint is provided
    start_epoch = 0
    if args.resume:
        checkpoint = trainer.load_checkpoint(args.resume)
        start_epoch = checkpoint.get('epoch', 0) + 1
        logger.info(f"Resuming training from epoch {start_epoch}")
    
    # Train model
    trainer.train(
        num_epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience
    )
    
    logger.info("Training completed")


if __name__ == "__main__":
    main()
