import csv
from datetime import datetime
from pathlib import Path
from typing import Generator, List
from collections import deque
from models.weather_data import WeatherData
from .utils import extract_city
import os


WeatherDataGenerator = Generator[List[WeatherData], None, None]

class FileReader:
    @staticmethod
    def read_csv(file_path: Path, batch_size: int = 1000) -> WeatherDataGenerator:
        encoding = 'latin-1'  # Tenta UTF-8 primeiro, depois Latin-1
        weather_data_list = deque(maxlen=batch_size)
        skip_lines = 8


        # Tenta abrir o arquivo com diferentes encodings
        try:
            with open(file_path, mode='r', encoding=encoding) as file:
                reader = csv.DictReader(file, delimiter=';')
                file_path = file.name
                print(f"file_path: {file_path}")

                # Extract the filename from the path
                filename = os.path.basename(file_path)
                print(f"filename: {filename}")

                
                city = extract_city(filename)
                print(f"city: {city}")

                for _ in range(skip_lines):
                    next(file)

                for row in reader:                        
                    if not row.get('Data'):
                        print("Não tem data...")
                        continue

                    # Iterando sobre os itens do dicionário
                    for chave, valor in row.items():
                        # Alterando o valor da chave
                        if type(valor) == str and valor.isnumeric():
                            row[chave] = valor.replace(',' , '.')

                    print(f"row: {len(row)}")

                    # print(meu_dict)

                    # Combine date and time into a single timestamp
                    reading_time = datetime.strptime(f"{row['Data']} {row['Hora UTC']}", "%Y/%m/%d %H%M UTC")
                    weather_data = WeatherData(
                        city=city,
                        reading_time=reading_time,
                        precipitation=float(row['PRECIPITAÇÃO TOTAL, HORÁRIO (mm)'].replace(',' , '.') or 0),
                        station_pressure=float(row['PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)'].replace(',' , '.') or 0),
                        max_pressure=float(row['PRESSÃO ATMOSFERICA MAX.NA HORA ANT. (AUT) (mB)'].replace(',' , '.') or 0),
                        min_pressure=float(row['PRESSÃO ATMOSFERICA MIN. NA HORA ANT. (AUT) (mB)'].replace(',' , '.') or 0),
                        global_radiation=float(row['RADIACAO GLOBAL (Kj/m²)'].replace(',' , '.') or 0),
                        temperature=float(row['TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)'].replace(',' , '.') or 0),
                        dew_point=float(row['TEMPERATURA DO PONTO DE ORVALHO (°C)'].replace(',' , '.') or 0),
                        max_temperature=float(row['TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)'].replace(',' , '.') or 0),
                        min_temperature=float(row['TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)'].replace(',' , '.') or 0),
                        max_dew_point=float(row['TEMPERATURA ORVALHO MAX. NA HORA ANT. (AUT) (°C)'].replace(',' , '.') or 0),
                        min_dew_point=float(row['TEMPERATURA ORVALHO MIN. NA HORA ANT. (AUT) (°C)'].replace(',' , '.') or 0),
                        max_humidity=float(row['UMIDADE REL. MAX. NA HORA ANT. (AUT) (%)'].replace(',' , '.') or 0),
                        min_humidity=float(row['UMIDADE REL. MIN. NA HORA ANT. (AUT) (%)'].replace(',' , '.') or 0),
                        humidity=float(row['UMIDADE RELATIVA DO AR, HORARIA (%)'].replace(',' , '.') or 0),
                        wind_direction=float(row['VENTO, DIREÇÃO HORARIA (gr) (° (gr))'].replace(',' , '.') or 0),
                        wind_gust=float(row['VENTO, RAJADA MAXIMA (m/s)'].replace(',' , '.') or 0),
                        wind_speed=float(row['VENTO, VELOCIDADE HORARIA (m/s)'].replace(',' , '.') or 0)
                    )
                    # print(f"weather_data: {weather_data}")

                    weather_data_list.append(weather_data)

                    if len(weather_data_list) == batch_size:
                        print(f"weather_data_list: {weather_data_list}")
                        yield list(weather_data_list)
                        weather_data_list.clear()

                # Yield any remaining data in the deque
                if weather_data_list:
                    yield list(weather_data_list)
            
        # except UnicodeDecodeError as uniErr:
        #     print(f"unicodeDecodeError: {uniErr}, trying another encoding...")
        #     continue  # Tenta o próximo encoding se houver erro de decodificação

        except Exception as e:
            print(f"Error reading file: {e}")
            raise e # Re-raise the exception to be handled by the caller

                