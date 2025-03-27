# Transformer Occupancy Prediction

A transformer-based model for predicting bus occupancy levels by classifying them into three categories: Empty (0), Half Full (1), and Full (2). The model uses temporal, spatial, and environmental features to make sequence-to-sequence predictions for bus stops along a route.

## Features

- Transformer architecture for sequence-to-sequence prediction
- Handles variable-length trip sequences through padding and masking
- Temporal feature encoding (hour, day)
- Spatial feature encoding (bus stop location)
- Environmental feature processing (weather conditions)
- Class imbalance handling through weighted loss
- Real-time occupancy prediction for each bus stop in a trip
- Comprehensive metrics visualization and analysis

## Quick Start
To quickly get started with training and running the model:

```bash
# Create a virtual environment
python -m venv venv
# Activate the virtual environment
source venv/bin/activate  # Linux/macOS
# Install dependencies
pip install -r requirements.txt
# Extract the example data
tar -xvjf data.tar.bz2
# Create a configuration file from the example
cp config-example.json config.json # Edit the configuration if needed
# Train the model
python train.py --config config.json
# Generate metric charts
python metrics_viz.py
# Run inference on test data
python inference.py --data data  --checkpoint checkpoints/best_model_acc.pt --config checkpoints/config.json --output output/test_inference.csv  --device cuda
```

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd transformer-occupancy-prediction
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies

- torch>=1.9.0
- numpy>=1.19.0
- tqdm>=4.62.0
- scikit-learn>=0.24.0
- matplotlib>=3.4.0
- pandas>=1.3.0

## Configuration

The model and training parameters can be configured using a JSON configuration file. Example configuration options in `config-example.json`:

```json
{
  "data_path": "data",
  "batch_size": 128,
  "num_workers": 4,
  "input_dim": 8,
  "hidden_dim": 1024,
  "nhead": 8,
  "num_encoder_layers": 6,
  "num_classes": 3,
  "dropout": 0.2,
  "use_positional_encoding": true,
  "max_seq_len": 100,
  "use_cls_token": true,
  "learning_rate": 0.00003,
  "weight_decay": 0.01
}
```

Copy `config-example.json` to create your own configuration:
```bash
cp config-example.json config.json
```

## Usage

### Training

Train the model using the training script with custom arguments:

```bash
python train.py \
  --data-path data \
  --batch-size 32 \
  --hidden-dim 128 \
  --learning-rate 0.001 \
  --epochs 50 \
  --device cuda
```

Key training arguments:
- `--data-path`: Path to training data directory
- `--batch-size`: Training batch size
- `--hidden-dim`: Hidden dimension size
- `--learning-rate`: Learning rate
- `--epochs`: Number of training epochs
- `--device`: Device to use (cuda/cpu)
- `--config`: Path to configuration file
- `--resume`: Path to checkpoint for resuming training

### Inference

Run inference on test data:

```bash
python inference.py \
  --data data \
  --checkpoint checkpoints/model.pt \
  --config config.json \
  --output predictions.csv \
  --include-probabilities
```

Key inference arguments:
- `--data`: Path to test dataset
- `--checkpoint`: Path to model checkpoint
- `--config`: Path to model configuration
- `--output`: Output file path for predictions
- `--include-probabilities`: Include class probabilities in output
- `--device`: Device to use (cuda/cpu)

## Development

1. Setup your development environment:
```bash
pip install -r requirements.txt
```

## Project Structure

- `train.py`: Main training script
- `inference.py`: Inference script for making predictions
- `model.py`: Model architecture implementation
- `feature_engineering.py`: Feature preprocessing and engineering
- `read_data.py`: Data loading utilities
- `iterable_occupancy_dataset.py`: PyTorch dataset implementation with iterable dataset
- `occupancy_dataset.py`: PyTorch dataset implementation
- `metrics_viz.py`: Visualization utilities for model metrics
- `metrics_calc.py`: Metrics calculation utilities
- `config-example.json`: Example configuration file
- `requirements.txt`: Project dependencies

## Model Architecture

The model uses a transformer-based architecture with:
- Input embedding layer
- Positional encoding
- 6 transformer encoder layers
- 8 attention heads
- Hidden dimension of 1024
- Dropout for regularization
- Classification head with ReLU activation

## Analysis Scripts

### Occupation Distribution Analysis

The `occupation_distribution_analysis.py` script provides visualizations of bus occupancy distributions across different time slots. It processes trip data from JSON Lines files (located by default in `data/**/*.jsonl`) and generates an interactive Plotly dashboard showing the percentage distribution of occupancy levels for each bus stop.

Features:
- Groups trips by time (HH:MM:SS portion of tripScheduledTime)
- Calculates percentage distribution of occupancyLevel per busStopLocation
- Generates an interactive dashboard with a dropdown menu for time slot selection
- Displays occupancy distributions using stacked bar charts
- Supports filtering by day of the week

Usage:
```bash
# Process all JSONL files in the default directory (data/output/)
python occupation_distribution_analysis.py

# Process specific JSONL files or directories
python occupation_distribution_analysis.py data/train data/test data/val

# Filter data by specific days (sun, mon, tue, wed, thu, fri, sat, weekdays, weekend)
python occupation_distribution_analysis.py --day_filter mon wed fri -- data/train data/test data/val
```

The script will open an interactive dashboard in your default web browser, allowing you to:
- Select different time slots from the dropdown menu
- View stacked bar charts showing occupancy distribution per bus stop
- Analyze occupancy patterns across different times and locations

## Contact

For questions and support, please open an issue in the repository's issue tracker.

## License

[Add License Information]

