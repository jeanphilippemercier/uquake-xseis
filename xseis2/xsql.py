from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import ARRAY

Base = declarative_base()


class DataFile(Base):
    __tablename__ = 'datafile'

    name = Column(String(50), primary_key=True)
    status = Column(Boolean)

    def __repr__(self):
        return f"<DataFile ({self.name}) [processed_status: {self.status}]>"


class Channel(Base):
    __tablename__ = 'channel'

    name = Column(String(10), primary_key=True)
    component = Column(String(5))
    # station = Column(String(10))
    station_name = Column(String(10), ForeignKey('station.name'))
    station = relationship("Station")
    quality = Column(Integer)
    samplerate = Column(Float)

    def __repr__(self):
        return f"<Channel ({self.name}) [samplerate: {self.samplerate} Hz] [quality: {self.quality}]>"


class Station(Base):
    __tablename__ = 'station'

    name = Column(String(10), primary_key=True)
    channels = Column(ARRAY(String(10)))
    location = Column(ARRAY(Float))

    def __repr__(self):
        str_loc = ", ".join([f"{v:.0f}" for v in self.location])
        str_chans = ",".join(self.channels)
        return f"<Station ({self.name}) [{str_chans}] [{str_loc}]>"


class StationPair(Base):
    __tablename__ = 'stationpair'

    name = Column(String(20), primary_key=True)
    station1_name = Column(String(10), ForeignKey('station.name'))
    station2_name = Column(String(10), ForeignKey('station.name'))
    station1 = relationship("Station", foreign_keys=[station1_name])
    station2 = relationship("Station", foreign_keys=[station2_name])
    dist = Column(Float)

    def __repr__(self):
        return f"<StationPair ({self.name}) [{self.dist:.2f} m]>"


class XCorr(Base):
    __tablename__ = 'xcorr'

    id = Column(Integer, primary_key=True)
    corr_key = Column(String(20))

    stationpair_name = Column(String(20), ForeignKey('stationpair.name'))
    stationpair = relationship("StationPair")

    channel1_name = Column(String(10), ForeignKey('channel.name'))
    channel2_name = Column(String(10), ForeignKey('channel.name'))
    channel1 = relationship("Channel", foreign_keys=[channel1_name])
    channel2 = relationship("Channel", foreign_keys=[channel2_name])

    start_time = Column(DateTime)
    length = Column(Float)  # length of stack in hours
    nstack = Column(Float)  # percentage of data kept
    waveform_redis_key = Column(String(50))  # redis key of cross-correlation waveform
    # delta_velocity = Column(Float)  # dv/v measurement versus reference trace
    # error = Column(Float)
    dvv = Column(ARRAY(Float))

    def __repr__(self):

        if self.dvv is None:
            dvv_str = None
        else:
            dvv_str = ", ".join([f"{v:.2f}" for v in self.dvv])

        return f"<Xcorr ({self.corr_key}) {self.start_time} [{self.stationpair.dist:.2f} m] [kept {self.nstack:.2f}% of {self.length:.2f} hours]  [dvv: {dvv_str}]"
        # return f"<Xcorr ({self.corr_key}) {self.start_time} [{self.stationpair.dist:.2f} m] [hours: {self.length:.2f}]  [dvv: {self.delta_velocity}, error: {self.error}]"


class ChanPair(Base):
    __tablename__ = 'chanpair'

    corr_key = Column(String(20), primary_key=True)
    station1 = Column(String(10))
    station2 = Column(String(10))
    inter_dist = Column(Float)

    def __repr__(self):
        return f"<ChanPair({self.corr_key}, {self.inter_dist:.2f} m"


class VelChange(Base):
    __tablename__ = 'velchange'

    id = Column(Integer, primary_key=True)
    station = Column(String(10))  # seismic station ID
    start_time = Column(DateTime)  # measurement start time
    length = Column(String(10))  # identifier of length (e.g 'hourly', 'daily')
    delta_velocity = Column(Float)  # dv/v measurement
    error = Column(Float)  # error on the dv/v measurement

    def __repr__(self):
        return f"<VelChange({self.start_time}, {self.length}, {self.station}, {self.delta_velocity}, {self.error}"



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

# class Parent(Base):
#     __tablename__ = 'parent'
#     id = Column(Integer, primary_key=True)
#     child_id = Column(Integer, ForeignKey('child.id'))
#     child = relationship("Child")

# class Child(Base):
#     __tablename__ = 'child'
#     id = Column(Integer, primary_key=True)
