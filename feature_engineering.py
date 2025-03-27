import math
from datetime import datetime
from typing import Union
import pytz

rush_hours_ranges = [(6 * 60, 9 * 60), (17 * 60 + 40, 20 * 60)]

def cyclical_encode(value, period):
    """
    Encode a cyclical feature (e.g., hour-of-day or day-of-week) using sine and cosine.

    Parameters:
        value (float): The value to encode.
        period (float): The period of the cycle (e.g., 24 for hours, 7 for days).

    Returns:
        tuple: (sin(value), cos(value))
    """
    if value < 0 or value >= period:
        raise ValueError(f"Value {value} is out of range for period {period}")

    angle = 2 * math.pi * value / period
    return math.sin(angle), math.cos(angle)


def normalize(value, min_val, max_val):
    """Normalize a value to the range [0, 1]."""
    return (value - min_val) / (max_val - min_val)


def standardize(value, mean, std):
    """Standardize a value to have zero mean and unit variance."""
    return (value - mean) / std


def process_record(record: dict, tz=pytz.timezone("America/Sao_Paulo")) -> dict:
    """Process a single trip record to extract and transform features
        the returned dict will have the following keys:
        - sin_hour
        - cos_hour
        - sin_day
        - cos_day
        - delay_minutes
        # - rush_hour
        # - normalized_day_of_week_kind
        - normalized_bus_stop_location
        - temperature
        - precipitation
        - trip_route_id
        - occupancy_level

    Args:
        record (dict): A dictionary representing a trip record.
        tz (pytz.timezone|str): A timezone object or string to use for datetime conversion.

    Returns:
        dict: A dictionary with the processed features.
    """

    tz = pytz.timezone(tz) if isinstance(tz, str) else tz

    # Convert ISO timestamps to timezone-aware datetime objects
    trip_scheduled_time = datetime.fromisoformat(
        record["tripScheduledTime"].replace("Z", "+00:00")
    ).astimezone(tz)
    trip_start_time = datetime.fromisoformat(
        record["tripStartTime"].replace("Z", "+00:00")
    ).astimezone(tz)

    # Extract cyclical features for hour-of-day
    hour = trip_scheduled_time.hour * 60 + trip_scheduled_time.minute
    sin_hour, cos_hour = cyclical_encode(hour, 24 * 60)

    # Extract cyclical features for day-of-week (Monday=0, Sunday=6)
    day_of_week = trip_scheduled_time.weekday()
    sin_day, cos_day = cyclical_encode(day_of_week, 7)

    # Calculate trip delay in minutes
    delay_minutes = round(
        (trip_start_time - trip_scheduled_time).total_seconds() / 60.0
    )

    # Normalize bus stop location
    route_length = record.get("routeTotalLength", 1)  # Avoid division by zero
    normalized_bus_stop_location = record["busStopLocation"] / route_length

    # Extract temperature and precipitation
    temp: float = record["weatherTemperature"]
    precip: float = record["weatherPrecipitation"]

    # Extract month and encode cyclically
    # month = trip_scheduled_time.month
    # sin_month, cos_month = cyclical_encode(month - 1, 12)

    # day_of_week_kind = 0 if day_of_week < 5 else 1 if day_of_week == 5 else 2 # 0: weekday, 1: saturday, 2: sunday
    # normalized_day_of_week_kind = normalize(day_of_week_kind, 0, 2)

    # rush_hour = 1 if  any(
    #     start <= hour <= end for start, end in rush_hours_ranges
    # ) else 0

    # sin_hour: scalar, cos_hour: scalar, sin_day: scalar, cos_day: scalar, delay_minutes: scalar, normalized_bus_stop_location: scalar, sin_month: scalar, cos_month: scalar, trip_route_id: embedding, temperature: float, precipitation: float, occupancy_level: int
    features = {
        "sin_hour": sin_hour,
        "cos_hour": cos_hour,
        "sin_day": sin_day,
        "cos_day": cos_day,
        # "sin_month": sin_month,
        # "cos_month": cos_month,
        # "rush_hour": rush_hour,
        # "normalized_day_of_week_kind": normalized_day_of_week_kind,
        "normalized_bus_stop_location": normalized_bus_stop_location,
        "delay_minutes": delay_minutes,
        "temperature": temp,
        "precipitation": precip,
        "trip_route_id": record["tripRouteId"],
        # Target: occupancy level
        "occupancy_level": record["occupancyLevel"],
    }

    return features


def standardize_record(processed_record, statistics):
    """Standardize the numerical features of a processed record using the provided statistics
        The returned dict will have the following keys
        - sin_hour
        - cos_hour
        - sin_day
        - cos_day
        # - sin_month
        # - cos_month
        # - rush_hour
        # - normalized_day_of_week_kind
        - normalized_bus_stop_location
        - delay_minutes_standardized
        - temperature_standardized
        - precipitation_standardized
        - trip_route_id
        - occupancy_level


    Args:
        processed_record (dict): A dictionary with the processed features as returned by process_record
        statistics (dict): A dictionary with the statistics for each feature, as returned by compute_statistics

    Returns:
        dict: A dictionary with the standardized features having the same keys as the input record except for delay_minutes, temperature, and precipitation which are replaced by delay_standardized, temperature_standardized, and precipitation_standardized
    """
    delay_minutes = processed_record["delay_minutes"]
    temp = processed_record["temperature"]
    precip = processed_record["precipitation"]
    delay_standardized = standardize(
        delay_minutes,
        statistics["delay_minutes"]["mean"],
        statistics["delay_minutes"]["std"],
    )
    temp_standardized = standardize(
        temp, statistics["temperature"]["mean"], statistics["temperature"]["std"]
    )
    precip_standardized = standardize(
        precip, statistics["precipitation"]["mean"], statistics["precipitation"]["std"]
    )

    standardized_record = {
        **{
            k: v
            for k, v in processed_record.items()
            if k not in ["delay_minutes", "temperature", "precipitation"]
        },
        "delay_minutes_standardized": delay_standardized,
        "temperature_standardized": temp_standardized,
        "precipitation_standardized": precip_standardized,
    }
    return standardized_record


def process_trip_records(
    trip_records, statistics=None, tz: Union[str, pytz.timezone] = "America/Sao_Paulo"
) -> list[dict]:
    """
    Process a list of trip records to extract and transform features.

    This function converts timestamps into timezone-aware datetime objects,
    extracts cyclical features for hour-of-day and day-of-week, calculates trip delay,
    normalizes the bus stop location, and converts weather data into categorical values.

    Parameters:
        trip_records (list): A list of dictionaries, each representing an occupancy record.
        statistics (dict): A dictionary with the statistics for each feature, as returned by compute_statistics.
        tz (str|pytz.timezone): A timezone object or string to use for datetime conversion.

    Returns:
        list: A list of dictionaries, each with the processed features. If statistics is provided, the features are standardized.
    """
    tz = pytz.timezone(tz) if isinstance(tz, str) else tz
    processed = []
    for record in trip_records:
        features = process_record(record, tz)
        if statistics is not None:
            features = standardize_record(features, statistics)
        processed.append(features)

    return processed
