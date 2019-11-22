from sqlalchemy import Column, Integer, String, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import ARRAY

Base = declarative_base()

# "results": [
#         {
#             "id": 1,
#             "sensor": {
#                 "id": 1,
#                 "code": "1"
#             },
#             "start_time": "2011-01-01T11:01:11Z",
#             "length": 111.0,
#             "delta_velocity": 11.0,
#             "error": 1.0
#         }
#     ]


class Station(Base):
    __tablename__ = 'stations'

    code = Column(String(10), primary_key=True)
    channels = Column(ARRAY(String(5)))
    location = Column(ARRAY(Float))
    quality = Column(Integer)

    def __repr__(self):
        str_loc = ", ".join([f"{v:.0f}" for v in self.location])
        str_chans = ",".join(self.channels)
        return f"<Station ({self.code}) [{str_chans}] [{str_loc}] [quality: {self.quality}]>"


class StationPair(Base):
    __tablename__ = 'station_pairs'

    code = Column(String(20), primary_key=True)
    station1 = Column(String(10))
    station2 = Column(String(10))
    dist = Column(Float)

    def __repr__(self):
        return f"<ChanPair ({self.code}) [{self.dist:.2f} m]>"


class ChanPair(Base):
    __tablename__ = 'chanpairs'

    corr_key = Column(String(20), primary_key=True)
    station1 = Column(String(10))
    station2 = Column(String(10))
    inter_dist = Column(Float)

    def __repr__(self):
        return f"<ChanPair({self.corr_key}, {self.inter_dist:.2f} m"


class VelChange(Base):
    __tablename__ = 'velchanges'

    id = Column(Integer, primary_key=True)
    station = Column(String(10))  # seismic station ID
    start_time = Column(DateTime)  # measurement start time
    length = Column(String(10))  # identifier of length (e.g 'hourly', 'daily')
    delta_velocity = Column(Float)  # dv/v measurement
    error = Column(Float)  # error on the dv/v measurement

    def __repr__(self):
        return f"<VelChange({self.start_time}, {self.length}, {self.station}, {self.delta_velocity}, {self.error}"


class XCorr(Base):
    __tablename__ = 'xcorrs'

    id = Column(Integer, primary_key=True)
    start_time = Column(DateTime)
    length = Column(String(10))  # identifier of length (e.g 'hourly', 'daily')
    corr_key = Column(String(20))
    waveform_redis_key = Column(String(50))  # redis key of cross-correlation waveform
    delta_velocity = Column(Float)  # dv/v measurement versus reference trace
    error = Column(Float)

    def __repr__(self):
        return f"<Xcorr({self.start_time}, {self.length}, {self.corr_key}, {self.waveform_redis_key}, {self.delta_velocity}, {self.error}"

