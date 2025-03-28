#!/usr/bin/env python3
"""
This script reads trip data from JSON Lines files located in the data/output folder,
groups trips by the time (HH:MM:SS portion of tripScheduledTime), and for each time slot,
calculates the percentage distribution of occupancyLevel per busStopLocation.
An interactive Plotly dashboard is generated with a dropdown menu to select the time slot.
Upon selecting a time, a stacked bar chart displays the occupancy distribution for each bus stop.

The data can be filtered by day of week using the --day_filter argument. Valid values are:
sun, mon, tue, wed, thu, fri, sat, weekdays, weekend
Multiple values can be provided to include multiple days.
"""

import argparse
import os
import json
import glob
import pandas as pd
import plotly.graph_objects as go

def read_trip_data(file_list):
    """Reads JSON Lines files and returns a DataFrame of trip data."""
    records = []
    for filepath in file_list:
        with open(filepath, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        records.append(record)
                    except json.JSONDecodeError:
                        continue
    return pd.DataFrame(records)

# Day filter mapping
DAY_FILTER_MAP = {
    'sun': 'Sunday',
    'mon': 'Monday',
    'tue': 'Tuesday',
    'wed': 'Wednesday',
    'thu': 'Thursday',
    'fri': 'Friday',
    'sat': 'Saturday',
    'weekdays': {'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'},
    'weekend': {'Saturday', 'Sunday'}
}

def get_jsonl_files(paths):
    """
    Given a list of file or directory paths, returns a list of JSONL files.
    If a path is a directory, its immediate children ending with '.jsonl' are added.
    """
    files = []
    for path in paths:
        if os.path.isfile(path) and path.endswith(".jsonl"):
            files.append(path)
        elif os.path.isdir(path):
            files.extend(glob.glob(os.path.join(path, "*.jsonl")))
    return files

def main():
    parser = argparse.ArgumentParser(
        description="Generate occupancy dashboard from JSONL trip data files"
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="File or directory paths to read JSONL data from. If a directory is provided, its .jsonl files will be used."
    )
    parser.add_argument(
        "--day_filter",
        nargs="+",
        choices=list(DAY_FILTER_MAP.keys()),
        help="Filter by day of week. Valid values: sun, mon, tue, wed, thu, fri, sat, weekdays, weekend. Multiple values allowed."
    )
    args = parser.parse_args()
    
    if args.paths:
        jsonl_files = get_jsonl_files(args.paths)
        if not jsonl_files:
            raise ValueError("No JSONL files found in provided paths.")
    else:
        jsonl_files = glob.glob("data/output/*.jsonl")
        if not jsonl_files:
            raise ValueError("No JSONL files found in default directory 'data/output/'.")
    
    # Read the data from all JSONL files
    df = read_trip_data(jsonl_files)

    # Ensure required columns exist
    required_fields = {"tripScheduledTime", "occupancyLevel", "busStopLocation"}
    if not required_fields.issubset(set(df.columns)):
        raise ValueError(f"Missing fields in trip data. Required fields: {required_fields}")

    # Parse tripScheduledTime to datetime and extract time and day
    try:
        df["datetime"] = pd.to_datetime(df["tripScheduledTime"])
        df["time"] = df["datetime"].dt.strftime("%H:%M:%S")
        df["day"] = df["datetime"].dt.day_name()
    except Exception as e:
        raise ValueError(f"Error parsing tripScheduledTime: {e}")

    # Apply day filter if provided
    if args.day_filter:
        # Create a set of allowed days based on the provided filters
        allowed_days = set()
        for day_filter in args.day_filter:
            days = DAY_FILTER_MAP[day_filter]
            if isinstance(days, str):
                allowed_days.add(days)
            else:
                allowed_days.update(days)
        
        # Filter the DataFrame to include only allowed days
        df = df[df["day"].isin(allowed_days)]
        
        if df.empty:
            raise ValueError("No data available for the specified day filter(s).")

    # Group the data: Count trips by time, busStopLocation, and occupancyLevel
    grp = df.groupby(["time", "busStopLocation", "occupancyLevel"]).size().reset_index(name="count")

    # Calculate percentage distribution per (time, busStopLocation)
    grp["total_per_bus_stop"] = grp.groupby(["time", "busStopLocation"])["count"].transform("sum")
    grp["percentage"] = grp["count"] / grp["total_per_bus_stop"] * 100

    # Pivot the data to have occupancyLevel as columns
    pivot_df = grp.pivot_table(index=["time", "busStopLocation"],
                               columns="occupancyLevel",
                               values="percentage",
                               fill_value=0).reset_index()

    # Ensure occupancy level columns are sorted and available across all time slots.
    occupancy_levels = sorted(df["occupancyLevel"].unique())
    color_mapping = {
        0: "#19BF1F",
        1: "#BF8619",
        2: "#BF2019"
    }

    # Get sorted unique time slots
    time_slots = sorted(df["time"].unique())
    if not time_slots:
        raise ValueError("No valid time slots found in data.")

    # Prepare initial data for the first time slot
    initial_time = time_slots[0]
    df_initial = pivot_df[pivot_df["time"] == initial_time]

    # Create the initial traces (one per occupancy level)
    traces = []
    for level in occupancy_levels:
        y = df_initial[level] if level in df_initial.columns else [0] * len(df_initial)
        traces.append(go.Bar(
            name=f"Occupancy {level}",
            x=df_initial["busStopLocation"],
            y=y,
            marker_color=color_mapping.get(level, "#000000")
        ))

    # Create the figure with the initial traces
    fig = go.Figure(data=traces)

    # Build frames for each time slot
    frames = []
    for t in time_slots:
        df_t = pivot_df[pivot_df["time"] == t]
        frame_data = []
        for level in occupancy_levels:
            y = df_t[level] if level in df_t.columns else [0] * len(df_t)
            frame_data.append(go.Bar(
                x=df_t["busStopLocation"],
                y=y,
                name=f"Occupancy {level}",
                marker_color=color_mapping.get(level, "#000000")
            ))
        frames.append(go.Frame(data=frame_data, name=t))
    fig.frames = frames

    # Create dropdown menu buttons to select time slot via animation frames
    dropdown_buttons = [
        {
            "label": t,
            "method": "animate",
            "args": [[t],
                     {"frame": {"duration": 500, "redraw": True},
                      "mode": "immediate",
                      "transition": {"duration": 300}}]
        } for t in time_slots
    ]

    # Update layout with dropdown menu and stacked bar mode
    fig.update_layout(
        title=f"Occupancy Distribution at {initial_time}",
        updatemenus=[{
            "buttons": dropdown_buttons,
            "direction": "down",
            "showactive": True,
            "x": 1.15,
            "y": 0.8,
            "xanchor": "left",
            "yanchor": "top"
        }],
        barmode="stack",
        xaxis={"title": "Bus Stop Location"},
        yaxis={"title": "Percentage (%)"}
    )

    # Display the figure. This will open in your default browser if not in a Jupyter notebook.
    fig.show()

if __name__ == "__main__":
    main()
