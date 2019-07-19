import os
from datetime import datetime, timedelta
import numpy as np

from sqlalchemy import Table, Column, Integer, String, MetaData, ForeignKey, DateTime, Float, Sequence, ARRAY, LargeBinary, Boolean
from sqlalchemy.sql import select
# from sqlalchemy import create_engine
# from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class VelChange(Base):
    __tablename__ = 'velchanges'

    id = Column(Integer, primary_key=True)
    time = Column(DateTime)
    sta = Column(String(10))
    # hours = Column(String(10))
    dvv = Column(Float)
    error = Column(Float)

    def __repr__(self):
        return f"<VelChange({self.time}, {self.sta}, {self.dvv}, {self.error}"


class XCorr(Base):
    __tablename__ = 'xcorrs'

    id = Column(Integer, primary_key=True)
    time = Column(DateTime)
    ckey = Column(String(20))
    data = Column(String(50))
    dvv = Column(Float)
    error = Column(Float)

    def __repr__(self):
        return f"<Xcorr({self.time}, {self.ckey}, {self.data}, {self.dvv}, {self.error}"


class ChanPair(Base):
    __tablename__ = 'chanpairs'

    ckey = Column(String(20), primary_key=True)
    sta1 = Column(String(10))
    sta2 = Column(String(10))
    dist = Column(Float)

    def __repr__(self):
        return f"<ChanPair({self.ckey}, {self.dist:.2f} m"
