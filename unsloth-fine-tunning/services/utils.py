import re
import argparse

from datetime import datetime, date
import pandas as pd
import os




def extract_city(filename: str) -> str:
  # pattern = r"INMET_SE_SP_A\d{3}_([A-Z]+)_\d{2}-\d{2}-\d{4}_A_\d{2}-\d{2}-\d{4}\.CSV"
  # pattern = r"INMET_SE_([A-Z]{2}+)_A\d{3}_([A-Z]+)_\d{2}-\d{2}-\d{4}_A_\d{2}-\d{2}-\d{4}\.CSV"
  pattern = r"INMET_[A-Z]{2}_[A-Z]{2}_A\d{3}_([^_]+)_\d{2}-\d{2}-\d{4}_A_\d{2}-\d{2}-\d{4}\.CSV"


  # Search for the pattern in the filename
  match = re.search(pattern, filename)

  # Extract the city name if a match is found
  return match.group(1) if match else None

def valid_date(date_str) -> date:
  try:
      return datetime.strptime(date_str, "%Y-%m-%d").date()
  except ValueError:
      raise argparse.ArgumentTypeError(f"Invalid date format: {date_str}. Use YYYY-MM-DD.")


def read_last_line(file_path):
  # Check until data was stored
  if os.path.exists(file_path):
    # Lê o arquivo CSV inteiro
    df = pd.read_csv(file_path)

    # Seleciona a última linha
    last_line = df.iloc[-1]

    return last_line
  
  return None
   

if __name__ == '__main__':
  # Example filename
  # filename = "INMET_SE_SP_A714_ITAPEVA_01-01-2024_A_31-12-2024.CSV"
  filename = "INMET_SE_RJ_A620_CAMPOS DOS GOYTACAZES - SAO TOME_01-01-2023_A_31-12-2023.CSV"
  city = extract_city(filename)

  print(f"city: {city}")
