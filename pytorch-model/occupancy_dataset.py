"""
Occupancy Dataset implementation.
"""

import os
import json
import logging
from typing import List, Tuple, Dict, Any
import glob
import math
from datetime import datetime

import torch
from torch.utils.data import Dataset
import numpy as np

from read_data import resolve_files_paths

logger = logging.getLogger(__name__)

class BusOccupancyDataset(Dataset):
    """
    Dataset for bus occupancy prediction.
    
    Reads and processes JSONL files with bus occupancy records.
    Groups records by tripId to create sequences for transformer input.
    """
    
    def __init__(
        self,
        data_path: str,
        transform=None,
        standardize_features=True,
        time_features=True
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to directory containing JSONL files
            transform: Optional transform to apply to features
            standardize_features: Whether to standardize numerical features
            time_features: Whether to extract cyclic time features
        """
        self.data_path = data_path
        self.transform = transform
        self.standardize_features = standardize_features
        self.time_features = time_features
        
        # Load and process data
        self.trips = self._load_data()
        logger.info(f"Loaded {len(self.trips)} trips from {data_path}")
        
        # Extract trip IDs for reference
        self.trip_ids = [trip_id for trip_id, _ in self.trips]
        
        # Compute feature statistics if standardizing
        if standardize_features:
            self._compute_feature_stats()
    
    def _load_data(self) -> List[Tuple[int, List[Dict[str, Any]]]]:
        """
        Load data from JSONL files and group by tripId.
        
        Returns:
            List of tuples (tripId, records) where records is a list of processed records
        """
        trips = {}
        self.max_seq_length = 0  
        
        # Find all JSONL files in data_path
        jsonl_files = resolve_files_paths(self.data_path, pattern="*.jsonl")
        logger.info(f"Found {len(jsonl_files)} JSONL files in {self.data_path}")
        
        # Process each file
        for file_path in jsonl_files:
            with open(file_path, 'r') as f:
                for line in f:
                    # Parse JSONL record
                    record = json.loads(line.strip())
                    
                    # Extract tripId
                    trip_id = record.get('tripId')
                    
                    # Initialize trip if not seen before
                    if trip_id not in trips:
                        trips[trip_id] = []
                    
                    # Add record to trip
                    trips[trip_id].append(record)
        
        # Sort records within each trip by busStopLocation
        for trip_id, records in trips.items():
            trips[trip_id] = sorted(records, key=lambda x: x.get('busStopLocation', 0))
            # Update maximum sequence length
            self.max_seq_length = max(self.max_seq_length, len(records))
            
        logger.info(f"Maximum sequence length in dataset: {self.max_seq_length}")
        # Convert to list of tuples
        return [(trip_id, records) for trip_id, records in trips.items()]
    
    def _compute_feature_stats(self):
        """
        Compute mean and standard deviation for numerical features for standardization.
        """
        # Features to standardize
        self.feature_stats = {
            'busStopLocation': {'sum': 0, 'sum_sq': 0, 'count': 0},
            'weatherTemperature': {'sum': 0, 'sum_sq': 0, 'count': 0},
            'weatherPrecipitation': {'sum': 0, 'sum_sq': 0, 'count': 0},
            'delay_minutes': {'sum': 0, 'sum_sq': 0, 'count': 0}
        }
        
        # Collect statistics
        for _, records in self.trips:
            for record in records:
                # Bus stop location
                bus_stop_location = record.get('busStopLocation', 0)
                route_total_length = record.get('routeTotalLength', 1)
                normalized_location = bus_stop_location / route_total_length if route_total_length > 0 else 0
                
                self.feature_stats['busStopLocation']['sum'] += normalized_location
                self.feature_stats['busStopLocation']['sum_sq'] += normalized_location ** 2
                self.feature_stats['busStopLocation']['count'] += 1
                
                # Weather temperature
                temp = record.get('weatherTemperature', 0)
                self.feature_stats['weatherTemperature']['sum'] += temp
                self.feature_stats['weatherTemperature']['sum_sq'] += temp ** 2
                self.feature_stats['weatherTemperature']['count'] += 1
                
                # Weather precipitation
                precip = record.get('weatherPrecipitation', 0)
                self.feature_stats['weatherPrecipitation']['sum'] += precip
                self.feature_stats['weatherPrecipitation']['sum_sq'] += precip ** 2
                self.feature_stats['weatherPrecipitation']['count'] += 1
                
                # Delay calculation
                scheduled_time = datetime.fromisoformat(record.get('tripScheduledTime', '').replace('Z', '+00:00'))
                actual_time = datetime.fromisoformat(record.get('timestamp', '').replace('Z', '+00:00'))
                delay_minutes = (actual_time - scheduled_time).total_seconds() / 60
                
                self.feature_stats['delay_minutes']['sum'] += delay_minutes
                self.feature_stats['delay_minutes']['sum_sq'] += delay_minutes ** 2
                self.feature_stats['delay_minutes']['count'] += 1
        
        # Compute mean and std for each feature
        for feature, stats in self.feature_stats.items():
            count = stats['count']
            if count > 0:
                mean = stats['sum'] / count
                variance = (stats['sum_sq'] / count) - (mean ** 2)
                std = max(math.sqrt(variance), 1e-6)  # Avoid division by zero
                
                self.feature_stats[feature]['mean'] = mean
                self.feature_stats[feature]['std'] = std
            else:
                self.feature_stats[feature]['mean'] = 0
                self.feature_stats[feature]['std'] = 1
        
        logger.info(f"Feature statistics: {self.feature_stats}")
    
    def _extract_features(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and preprocess features from a record.
        
        Args:
            record: Raw record from JSONL file
            
        Returns:
            Dictionary with preprocessed features
        """
        features = {}
        
        # Extract timestamp components
        scheduled_time = datetime.fromisoformat(record.get('tripScheduledTime', '').replace('Z', '+00:00'))
        hour = scheduled_time.hour + scheduled_time.minute / 60
        day_of_week = scheduled_time.weekday()  # 0 = Monday, 6 = Sunday
        
        # Cyclic time encodings
        if self.time_features:
            features['sin_hour'] = math.sin(2 * math.pi * hour / 24)
            features['cos_hour'] = math.cos(2 * math.pi * hour / 24)
            features['sin_day'] = math.sin(2 * math.pi * day_of_week / 7)
            features['cos_day'] = math.cos(2 * math.pi * day_of_week / 7)
        
        # Normalize bus stop location
        bus_stop_location = record.get('busStopLocation', 0)
        route_total_length = record.get('routeTotalLength', 1)
        features['normalized_bus_stop_location'] = bus_stop_location / route_total_length if route_total_length > 0 else 0
        
        # Calculate delay
        actual_time = datetime.fromisoformat(record.get('timestamp', '').replace('Z', '+00:00'))
        delay_minutes = (actual_time - scheduled_time).total_seconds() / 60
        features['delay_minutes'] = delay_minutes
        
        # Weather features
        features['temperature'] = record.get('weatherTemperature', 0)
        features['precipitation'] = record.get('weatherPrecipitation', 0)
        
        # Standardize numerical features if required
        if self.standardize_features:          
            features['delay_minutes_standardized'] = (
                features['delay_minutes'] - 
                self.feature_stats['delay_minutes']['mean']
            ) / self.feature_stats['delay_minutes']['std']
            
            features['temperature_standardized'] = (
                features['temperature'] - 
                self.feature_stats['weatherTemperature']['mean']
            ) / self.feature_stats['weatherTemperature']['std']
            
            features['precipitation_standardized'] = (
                features['precipitation'] - 
                self.feature_stats['weatherPrecipitation']['mean']
            ) / self.feature_stats['weatherPrecipitation']['std']
        
        # Target variable
        features['occupancy_level'] = record.get('occupancyLevel', 0)
        
        return features
    
    def __len__(self):
        """Return the number of trips in the dataset."""
        return len(self.trips)
    
    def __getitem__(self, idx):
        """
        Get a single trip with all its records.
        
        Args:
            idx: Index of the trip
            
        Returns:
            Tuple of (features, targets) for the trip
        """
        trip_id, records = self.trips[idx]
        
        # Process each record in the trip
        processed_features = []
        processed_targets = []
        
        for record in records:
            # Extract features from record
            features = self._extract_features(record)
            
            # Create feature array in the order expected by the model
            feature_array = [
                features['sin_hour'],
                features['cos_hour'],
                features['sin_day'],
                features['cos_day'],
                features['normalized_bus_stop_location'],
                features['delay_minutes_standardized'],
                features['temperature_standardized'],
                features['precipitation_standardized']
            ]
            
            # Add to lists
            processed_features.append(feature_array)
            processed_targets.append(features['occupancy_level'])
        
        # Apply transformations if any
        if self.transform:
            processed_features = self.transform(processed_features)
        
        return processed_features, processed_targets