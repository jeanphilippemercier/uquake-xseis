from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import ARRAY

Base = declarative_base()


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
        return f"<Channel ({self.name}) " \
               f"[samplerate: {self.samplerate} Hz] " \
               f"[quality: {self.quality}]>"


class Station(Base):
    __tablename__ = 'station'

    name = Column(String(10), primary_key=True)
    channels = Column(ARRAY(String(10)))
    location = Column(ARRAY(Float))

    def __repr__(self):
        str_loc = ", ".join([f"{v:.0f}" for v in self.location])
        str_chans = ",".join(self.channels)
        return f"<Station ({self.name}) [{str_chans}] [{str_loc}]>"


class ChanPair(Base):
    __tablename__ = 'chanpair'

    name = Column(String(20), primary_key=True)
    stationpair_name = Column(String(20), ForeignKey('stationpair.name'))
    stationpair = relationship("StationPair", lazy='select')
    dist = Column(Float)

    def __repr__(self):
        return f"<ChanPair ({self.name}, [{self.dist:.2f} m]>"


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

    start_time = Column(DateTime)
    length = Column(Float)  # length of stack in hours
    nstack = Column(Float)  # percentage of data kept
    waveform_redis_key = Column(String(50))  # redis key of cross-correlation waveform
    samplerate = Column(Float)

    # status 0: unprocessed chunk  1: processed chunk  2: stacked (used for dvv)
    status = Column(Integer)
    velocity_change = Column(Float)  # dv/v measurement versus reference trace
    error = Column(Float)
    pearson = Column(Float)  # pearson coeff with reference dvv trace (coda only)

    def __repr__(self):

        return f"<Xcorr ({self.corr_key}) {self.start_time} " \
               f"[hours: {self.length:.2f}] [status: {self.status}] " \
               f"[dvv: {self.velocity_change}, error: {self.error}]"


class StationDvv(Base):
    __tablename__ = 'stationdvv'

    id = Column(String(40), primary_key=True)
    # id = Column(Integer, primary_key=True)
    station = Column(String(10))  # seismic station ID
    start_time = Column(DateTime)  # measurement start time
    length = Column(Float)  # length of stack in hours
    # length = Column(String(10))  # identifier of length (e.g 'hourly', 'daily')
    velocity_change = Column(Float)  # dv/v measurement
    error = Column(Float)  # error on the dv/v measurement
    navg = Column(Integer)  # number of chanpair measurements averaged

    def __repr__(self):
        return f"<StationDvv ({self.station}) {self.start_time} [hours: {self.length:.2f}]  [dvv: {self.velocity_change:.4f}, error: {self.error:.2e}, navg: {self.navg}]>"


# class DataFile(Base):
#     __tablename__ = 'datafile'

#     name = Column(String(50), primary_key=True)
#     status = Column(Boolean)

#     def __repr__(self):
#         return f"<DataFile ({self.name}) [processed_status: {self.status}]>"



# GET /api/v1/inventory/noise_correlations
# HTTP 200 OK
# Allow: GET, POST, HEAD, OPTIONS
# Content-Type: application/json
# Vary: Accept

# {
#     "next": null,
#     "previous": null,
#     "cursor_next": null,
#     "cursor_previous": null,
#     "current_page": 1,
#     "total_pages": 1,
#     "count": 1,
#     "results": [
#         {
#             "id": 1,
#             "sensor": {
#                 "id": 1,
#                 "code": "1"
#             },
#             "start_time": "2011-01-01T11:01:11Z",
#             "length": 111.0,
#             "velocity_change": 11.0,
#             "error": 1.0
#         }
#     ]
# }
    # corr_key = Column(String(20), ForeignKey('chanpair.corr_key'))
    # chan_pair = Column(String(20))

    # stationpair_name = Column(String(20))
    # stationpair_name = Column(String(20), ForeignKey('stationpair.name'))
    # stationpair = relationship("StationPair")

    # channel1_name = Column(String(10), ForeignKey('channel.name'))
    # channel2_name = Column(String(10), ForeignKey('channel.name'))
    # channel1 = relationship("Channel", foreign_keys=[channel1_name])
    # channel2 = relationship("Channel", foreign_keys=[channel2_name])
