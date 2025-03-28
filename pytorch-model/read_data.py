import glob
import json
import os
import random
from typing import Union

import torch


def resolve_files_paths(data_source, pattern=None)->list[str]:
    """ Resolve a list of file paths from a
    data source, which can be a directory path
    or a list of file paths.

    Args:
        data_source (Union[str, list[str]]): The data source.

    Returns:
        list: A list of file paths.
    """
    files = None
    if isinstance(data_source, list):
        files = []
        for source in data_source:
            files.extend(resolve_files_paths(source))
    elif isinstance(data_source, str):
        # Check if data_source is a directory path
        if os.path.isdir(data_source):
            files = glob.glob(os.path.join(data_source, pattern or "*"))
        else:
            # Assume it's a single file path
            files = [data_source]
    return files

def read_file_records(file_path, randomize: Union[bool, random.Random] = False):
    """
    Read a JSONL file and return its contents as a list of dictionaries.

    Parameters:
        file_path (str): Path to the JSONL file.
        randomize (Union[bool, random.Random]): If True, randomizes the lines.
                                               If a random.Random instance, uses it for shuffling.

    Returns:
        list: List of dictionaries, each representing a record from the JSONL file.
    """
    with open(file_path, "r") as f:
        lines = f
        if randomize:
            lines = list(lines)
            if isinstance(randomize, random.Random):
                randomize.shuffle(lines)
            else:
                random.shuffle(lines)

        for line in lines:
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in {file_path}: {e}")
                continue

def read_files_records(
    data_path: Union[list[str], str],
    pattern="*.jsonl",
    randomize_files: Union[bool, random.Random] = False,
    exclude_trip_ids=[],
):
    """
    Generator that yields records from JSONL files.

    Parameters:
        data_path (str, optional): Can be a directory or file path or a list.
        pattern (str): File naming pattern for JSONL files. Used only when data_dir is provided.
        randomize_files (Union[bool, random.Random]): If True, randomizes the files.
                                               If a random.Random instance, uses it for shuffling.
        exclude_trip_ids : list
            List of trip IDs to exclude from the generator

    Yields:
        record: A record from the JSONL file.

    Raises:
        ValueError: If neither data_dir nor file_list is provided,
                   or if both are provided.
    """

    file_paths = resolve_files_paths(data_path, pattern=pattern)

    # Randomize file order if requested
    if randomize_files:
        if isinstance(randomize_files, random.Random):
            randomize_files.shuffle(file_paths)
        else:
            random.shuffle(file_paths)

    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:  # Multi-process data loading
        worker_id = worker_info.id
        num_workers = worker_info.num_workers
        file_paths = file_paths[worker_id::num_workers]

    for file_path in file_paths:
        records_iterator = read_file_records(file_path, randomize=False)
        try:
            for record in records_iterator:
                new_trip_id = record.get("tripId")
                if new_trip_id in exclude_trip_ids:
                    continue

                yield record
        finally:
            records_iterator.close()

def read_files_trips(
    data_path: Union[list[str], str],
    pattern="*.jsonl",
    randomize_files: Union[bool, random.Random] = False,
    randomize_records: Union[bool, random.Random] = False,
    exclude_trip_ids=[],
):
    """
    Generator that yields complete trip groups from JSONL files.

    Each group contains records with the same tripId, ordered by its location over the route LineString.

    Parameters:
        data_path (str, optional): Can be a directory or file path or a list.
        pattern (str): File naming pattern for JSONL files. Used only when data_dir is provided.
        randomize_files (Union[bool, random.Random]): If True, randomizes the files.
                                               If a random.Random instance, uses it for shuffling.
        randomize_records (Union[bool, random.Random]): If True, randomizes the records within each trip.
                                               If a random.Random instance, uses it for shuffling.
        exclude_trip_ids : list
            List of trip IDs to exclude from the generator

    Yields:
        tuple: (tripId, list of records for that trip)

    Raises:
        ValueError: If neither data_dir nor file_list is provided,
                   or if both are provided.
    """

    file_paths = resolve_files_paths(data_path, pattern=pattern)

    # Randomize file order if requested
    if randomize_files:
        if isinstance(randomize_files, random.Random):
            randomize_files.shuffle(file_paths)
        else:
            random.shuffle(file_paths)

    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:  # Multi-process data loading
        worker_id = worker_info.id
        num_workers = worker_info.num_workers
        file_paths = file_paths[worker_id::num_workers]

    for file_path in file_paths:
        current_trip_id = None
        trip_records = []
        records_iterator = read_file_records(file_path, randomize=False)
        try:
            for record in records_iterator:
                new_trip_id = record.get("tripId")
                if new_trip_id in exclude_trip_ids:
                    continue

                if current_trip_id is None:
                    current_trip_id = new_trip_id

                if new_trip_id != current_trip_id:
                    if randomize_records:
                        if isinstance(randomize_records, random.Random):
                            randomize_records.shuffle(trip_records)
                        else:
                            random.shuffle(trip_records)
                    yield current_trip_id, trip_records
                    current_trip_id = new_trip_id
                    trip_records = []

                trip_records.append(record)
            if trip_records:
                if randomize_records:
                    if isinstance(randomize_records, random.Random):
                        randomize_records.shuffle(trip_records)
                    else:
                        random.shuffle(trip_records)
                yield current_trip_id, trip_records
        finally:
            records_iterator.close()

def load_statistics(stats_file):
    with open(stats_file, 'r') as f:
        statistics = json.load(f)
    return statistics

def load_route_id_mapping(route_id_mapping_file):
    with open(route_id_mapping_file, 'r') as f:
        route_id_mapping = json.load(f)
    return route_id_mapping