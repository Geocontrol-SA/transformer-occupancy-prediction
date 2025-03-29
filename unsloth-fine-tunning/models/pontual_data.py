# models/weather_data.py
from dataclasses import dataclass
from datetime import datetime
from shapely import LineString, Point, wkb, from_wkt

@dataclass
class VehicleRunData:
    itinerary_id: int
    itinerary_code: str
    scheduledtime: datetime
    tripstarttime: datetime
    tripcompletiontime: datetime
    vehicle_id: int

    def __init__(self, itinerary_id, itinerary_code, scheduledtime, tripstarttime, tripcompletiontime, vehicle_id):
        self.itinerary_id = itinerary_id
        self.itinerary_code = itinerary_code
        self.scheduledtime = scheduledtime
        self.tripstarttime = tripstarttime
        self.tripcompletiontime = tripcompletiontime
        self.vehicle_id = vehicle_id
    

    
@dataclass
class BusStop:
    id: int
    code: str
    geo: Point


@dataclass
class BusOccupation:
    id: int
    vehicle_id: int
    reading_time: datetime
    occupation: int
    geo: Point


    def __init__(self,
            id: int,
            vehicle_id: int,
            reading_time: datetime,
            occupation: int,
            geo: str
            
        ):
        self.vehicle_id = vehicle_id
        self.reading_time = reading_time
        self.occupation = occupation
        self.geo = from_wkt(geo)






@dataclass
class ItineraryBusstopAssociation:
    id: int
    itinerary_id: int
    busstop_id: int
    busstop_code: str
    busstop_location: Point
    location: int

    def __init__(self, 
                id: int, 
                itinerary_id: int, 
                busstop_id: int, 
                busstop_code: str,                
                location: int,
                busstop_location: str
                ):
        
        

        self.id = id
        self.itinerary_id = itinerary_id
        self.busstop_id = busstop_id
        self.busstop_code = busstop_code
        self.busstop_location = from_wkt(busstop_location)
        self.location = location

    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        return self.id == other.id




@dataclass
class Itinerary:
    id: int
    code: str
    geo: LineString

    def __init__(self, id: int, code : str, geo: str):
        # print(f"geometry: {geo}")
        # print(f"type(geometry): {type(geo)}")
        # if isinstance(geometry, memoryview):
        #     geometry = bytes(geometry)
        self.id = id
        self.code = code
        # self.geo = wkb.load(geo, hex=True)
        self.geo = from_wkt(geo)
        # print(f"self.geo: {type(self.geo)}")
        self.itinerary_busstop_associations : list[ItineraryBusstopAssociation] = []

    
    def add_itinerary_busstop_association(self, association: ItineraryBusstopAssociation):
        self.itinerary_busstop_associations.append(association)
        sorted(self.itinerary_busstop_associations, key=lambda x: x.location)



    def __len__(self):
        if self.itinerary_busstop_associations:
            bustop_association = self.itinerary_busstop_associations[-1]
            return bustop_association.location
        else:
            return 0



    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id
    

from dataclasses import dataclass
from datetime import datetime
from shapely.geometry import Point

@dataclass
class VehiclerunBusStopoccupation:
    # vehiclerun data
    itinerary_code: str
    itinerary_size: int
    scheduled_time: str
    trip_start_time: str
    trip_completion_time: str
    vehicle_id: int

    # busstop_occupation_data
    busstop_code: str
    bustop_location: int
    reading_time: datetime
    occupation: int
    occupation_geo: Point
    occupation_location: int
    normalized_location: float

    temperature: float
    humidity: float
    wind_speed: float
    precipitation: float


    def __init__(self,
                 itinerary: 'Itinerary',
                 vehiclerun: 'VehicleRunData',
                 bus_occupation: 'BusOccupation',
                 association: 'ItineraryBusstopAssociation',
                 occupation_location: int
                 ):
        self.itinerary_code = itinerary.code
        self.itinerary_size = len(itinerary)
        self.scheduled_time = vehiclerun.scheduledtime
        self.trip_start_time = vehiclerun.tripstarttime
        self.trip_completion_time = vehiclerun.tripcompletiontime
        self.vehicle_id = vehiclerun.vehicle_id
        self.busstop_code = association.busstop_code
        self.bustop_location = association.location
        self.reading_time = bus_occupation.reading_time
        self.occupation = bus_occupation.occupation
        self.occupation_geo = bus_occupation.geo
        self.occupation_location = occupation_location
        self.normalized_location = self.bustop_location / self.itinerary_size
        self.temperature = None
        self.humidity = None
        self.wind_speed = None
        self.precipitation = None

    def to_dict(self):
        return {
            'itinerary_code': self.itinerary_code,
            'itinerary_size': self.itinerary_size,
            'scheduled_time': self.scheduled_time,
            'trip_start_time': self.trip_start_time,
            'trip_completion_time': self.trip_completion_time,
            'vehicle_id': self.vehicle_id,
            'busstop_code': self.busstop_code,
            'bustop_location': self.bustop_location,
            'reading_time': self.reading_time.isoformat(),  # Convert datetime to string
            'occupation': self.occupation,
            # 'occupation_geo': self.occupation_geo.wkt,  # Convert Point to Well-Known Text format
            'occupation_location': self.occupation_location,
            'normalized_location': self.normalized_location,
            'temperature': self.temperature,
            'humidity': self.humidity,
            'wind_speed': self.wind_speed,
            'precipitation': self.precipitation
        
        }
    



    
    