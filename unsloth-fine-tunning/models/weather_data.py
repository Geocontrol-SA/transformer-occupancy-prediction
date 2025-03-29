# models/weather_data.py
from dataclasses import dataclass
from datetime import datetime

@dataclass
class WeatherData:
    city: str
    reading_time: datetime
    precipitation: float
    station_pressure: float
    max_pressure: float
    min_pressure: float
    global_radiation: float
    temperature: float
    dew_point: float
    max_temperature: float
    min_temperature: float
    max_dew_point: float
    min_dew_point: float
    max_humidity: float
    min_humidity: float
    humidity: float
    wind_direction: float
    wind_gust: float
    wind_speed: float

    def __init__(self, 
                 
                city: str = None,
                reading_time: datetime = None,
                precipitation: float = None,
                station_pressure: float = None,
                max_pressure: float = None,
                min_pressure: float = None,
                global_radiation: float = None,
                temperature: float = None,
                dew_point: float = None,
                max_temperature: float = None,
                min_temperature: float = None,
                max_dew_point: float = None,
                min_dew_point: float = None,
                max_humidity: float = None,
                min_humidity: float = None,
                humidity: float = None,
                wind_direction: float = None,
                wind_gust: float = None,
                wind_speed: float = None

                ):
        self.city = city
        self.reading_time = reading_time
        self.precipitation = precipitation
        self.station_pressure = station_pressure
        self.max_pressure = max_pressure
        self.min_pressure = min_pressure
        self.global_radiation = global_radiation
        self.temperature = temperature
        self.dew_point = dew_point
        self.max_temperature = max_temperature
        self.min_temperature = min_temperature
        self.max_dew_point = max_dew_point
        self.min_dew_point = min_dew_point
        self.max_humidity = max_humidity
        self.min_humidity = min_humidity
        self.humidity = humidity
        self.wind_direction = wind_direction
        self.wind_gust = wind_gust
        self.wind_speed = wind_speed

        

    def to_dict(self):
        return {
            "city": self.city,
            "reading_time": self.reading_time,
            "precipitation": self.precipitation,
            "station_pressure": self.station_pressure,
            "max_pressure": self.max_pressure,
            "min_pressure": self.min_pressure,
            "global_radiation": self.global_radiation,
            "temperature": self.temperature,
            "dew_point": self.dew_point,
            "max_temperature": self.max_temperature,
            "min_temperature": self.min,
            "max_dew_point": self.max_dew_point,
            "min_dew_point": self.min_dew_point,
            "max_humidity": self.max_humidity,
            "min_humidity": self.min_humidity,
            "humidity": self.humidity,
            "wind_direction": self.wind_direction,
            "wind_gust": self.wind_gust,
            "wind_speed": self.wind_speed,

        }