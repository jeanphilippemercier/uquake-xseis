from importlib import reload
import os
import numpy as np
import time
# import h5py
from glob import glob
from datetime import datetime, timedelta

import redis
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xseis2 import xplot
from xseis2 import xutil
from xseis2 import xchange
from xseis2 import xchange_workflow as flow
from xseis2.xsql import Base, Channel, Station, StationPair, StationDvv, XCorr, ChanPair, DataFile
# from sqlalchemy.sql import exists

import itertools

from loguru import logger
from obspy import UTCDateTime
# from pytz import utc
import matplotlib.pyplot as plt
import pickle

from microquake.core.settings import settings

plt.ion()

#######################################################
#  xcorr processing params
#######################################################

params = settings.COMPUTE_XCORRS
print(params)

pair_dist_min = params.pair_dist_min
pair_dist_max = params.pair_dist_max
bad_chans = np.array(params.channel_blacklist)

#######################################################
#  clear databases
#######################################################
logger.info('Flushing redis database')
rhandle = redis.Redis(host='localhost', port=6379, db=0)
rhandle.flushall()
##################################

logger.info('Clearing psql tables')
db_string = "postgres://postgres:postgres@localhost"
db = create_engine(db_string)
print(db.table_names())
session = sessionmaker(bind=db)()
flow.sql_drop_tables(db)
session.commit()
Base.metadata.create_all(db)

#######################################################
#  create/fill sql tables
#######################################################

logger.info('Connect to redis database')

# with open(os.path.join(os.environ['SPP_COMMON'], "stations.pickle"), 'rb') as f:
#     stations = pickle.load(f)
stations = settings.inventory.stations()

flow.fill_table_stations(stations, session)
flow.fill_table_station_pairs(session)
flow.fill_table_chanpairs(session, pair_dist_min=pair_dist_min, pair_dist_max=pair_dist_max, bad_chans=bad_chans)

# stations = session.query(Station).all()
