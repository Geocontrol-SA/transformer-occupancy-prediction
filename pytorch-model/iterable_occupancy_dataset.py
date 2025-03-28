"""
Iterable Occupancy Dataset implementation.
"""

import json
import logging
import os
from torch.utils.data import IterableDataset
from feature_engineering import process_trip_records
from read_data import load_route_id_mapping, load_statistics, read_files_trips

logger = logging.getLogger(__name__)

class BusOccupancyDataset(IterableDataset):
    """
    Iterable dataset for bus occupancy prediction.
    Yield (features, targets) for each trip in the dataset.
    """

    def __init__(
        self,
        data_path: str,
        stats_file: str,
        route_id_mapping_file: str,
        trip_count: int,
        record_count: int,
        max_trip_records_len: int,
        transform=None,
        randomize_files=True,
        randomize_records=False,
    ):
        """
        Initialize the dataset.

        Args:
            data_path: Path to directory containing JSONL files
            stats_file: Path to statistics file
            route_id_mapping_file: Path to trip route ID mapping file
            trip_count: Number of trips in the dataset
            record_count: Number of records in the dataset
            max_trip_records_len: Maximum number of records in a trip
            transform: Optional transform to apply to features
        """
        self.data_path = data_path
        self.transform = transform
        self.trip_count = trip_count
        self.record_count = record_count
        self.max_seq_length = max_trip_records_len
        self.randomize_files = randomize_files
        self.randomize_records = randomize_records

        self.stats_file = stats_file
        # statistics about some features
        self.statistics = load_statistics(stats_file)

        # Load the trip_route_id mapping
        # This is used to map the trip_route_id to a normalized value starting from 0 up to the number of unique trip_route_ids
        self.route_id_mapping_file = route_id_mapping_file
        self.trip_route_id_mapping = load_route_id_mapping(route_id_mapping_file)
        self.num_trip_route_ids = len(self.route_id_mapping_file)

    def __len__(self):
        """
        Return the number of trips in the dataset.
        """
        return self.trip_count

    def __iter__(self):
        """
        Create an iterator that yields (features, targets) for each trip.
        """
        for trip_id, records in read_files_trips(
            self.data_path, randomize_files=self.randomize_files, randomize_records=self.randomize_records
        ):
            # Process each record in the trip
            processed_features = []
            processed_targets = []

            processed_records = process_trip_records(records, self.statistics)

            for record in processed_records:
                # trip_route_id_value = record.get("trip_route_id", "-1")
                # mapped_route_id = self.trip_route_id_mapping.get(
                #     str(trip_route_id_value)
                # )

                # Create feature array in the order expected by the model
                feature_array = [
                    record["sin_hour"],
                    record["cos_hour"],
                    record["sin_day"],
                    record["cos_day"],
                    # record["sin_month"],
                    # record["cos_month"],
                    # record["rush_hour"],
                    # record["normalized_day_of_week_kind"],
                    record["normalized_bus_stop_location"],
                    record["delay_minutes_standardized"],
                    record["temperature_standardized"],
                    record["precipitation_standardized"],
                ]

                # Add to lists
                processed_features.append(feature_array)
                processed_targets.append(record["occupancy_level"])

            # Apply transformations if any
            if self.transform:
                processed_features = self.transform(processed_features)

            yield processed_features, processed_targets

def create_dataset_from_processed_dir(dir: str, batch_name: str, transform=None, randomize_files=True, randomize_records=False):
    """
    Create a dataset from a directory containing preprocessed trip records.

    Args:
        dir: Directory containing data resulting of the process_files.py script
        batch_name: Name of the batch to be used for the dataset: train, val, test

    Returns:
        BusOccupancyDataset: Dataset instance
    """
    split_result_file = os.path.join(dir, "split_result.json")
    stats_file = os.path.join(dir, "stats.json")
    route_id_mapping_file = os.path.join(dir, "trip_route_id_mapping.json")
    jsonl_dir = os.path.join(dir, batch_name)
    if not os.path.exists(split_result_file):
        raise ValueError(f"split_result.json not found in {dir}")
    if not os.path.exists(stats_file):
        raise ValueError(f"stats.json not found in {dir}")
    if not os.path.exists(route_id_mapping_file):
        raise ValueError(f"trip_route_id_mapping.json not found in {dir}")
    if not os.path.exists(jsonl_dir):
        raise ValueError(f"{batch_name} directory not found in {dir}")
    
    with open(split_result_file, "r") as f:
        split_result = json.load(f)
    
    return BusOccupancyDataset(data_path=jsonl_dir, stats_file=stats_file, route_id_mapping_file=route_id_mapping_file, trip_count=split_result["trip_count"][batch_name], record_count=split_result["record_count"][batch_name], max_trip_records_len=split_result["max_trip_records_len"], transform=transform, randomize_files=randomize_files, randomize_records=randomize_records), split_result