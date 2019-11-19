from importlib import reload
import os
import numpy as np
import os
import time
# import h5py
# from glob import glob
from datetime import datetime, timedelta

import redis
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xseis2 import xplot
from xseis2 import xutil
from xseis2 import xchange
from xseis2 import xchange_workflow as flow
from xseis2.xsql import VelChange, XCorr, ChanPair, Base

from loguru import logger
from obspy import UTCDateTime
# from microquake.core.settings import settings
# from microquake.core.stream import Trace, Stream
# from pytz import utc
from microquake.plugin.site.core import read_csv
import matplotlib.pyplot as plt

plt.ion()

filename = "/home/phil/data/oyu/spp_common/sensors.csv"
inv = read_csv(filename, site_code='', has_header=True)
stations = inv.stations()
stations = sorted(stations, key=lambda x: x.code)


################################
logger.info('Connect to psql database')

db_string = "postgres://postgres:postgres@localhost"
db = create_engine(db_string)
print(db.table_names())
# Session = sessionmaker(bind=db)
# session = Session()
# session.close()
session = sessionmaker(bind=db)()
flow.sql_drop_tables(db)
session.commit()

Base.metadata.create_all(db)
session.commit()


##################################
logger.info('Connect to redis database')

rhandle = redis.Redis(host='localhost', port=6379, db=0)
rhandle.flushall()
###################################

dsr = 1000.0
# PARAMS - config
whiten_freqs = np.array([60, 80, 320, 350])
cc_wlen_sec = 20.0
stepsize_sec = cc_wlen_sec
# stepsize_sec = cc_wlen_sec / 2
keeplag_sec = 1.0
stacklen_sec = 600.0
onebit = True
min_pair_dist = 100
max_pair_dist = 800

####################################

curtime = datetime.utcnow()
# endtime = curtime + timedelta(minutes=-120)
# starttime = endtime - timedelta(minutes=10)
stacklen_sec = 600
times = [curtime + timedelta(seconds=i * stacklen_sec) for i in range(5)]
start_time = times[0]
end_time = start_time + timedelta(seconds=stacklen_sec)

# for t0 in times:
# print(t0)
# t1 = t0 + timedelta(seconds=stacklen_sec)

###############################

reload(flow)
flow.fill_table_chanpairs(stations, session, min_pair_dist, max_pair_dist)
# session.query(ChanPair).delete()
# flow.fill_table_chanpairs(stations, session)
# ckeys = np.unique(session.query(XCorr.corr_key).all())
cpairs = session.query(ChanPair).filter(ChanPair.inter_dist < max_pair_dist).all()
# req = session.query(ChanPair.corr_key).all()
# valid_corr_keys = np.array([x[0] for x in req])

valid_corr_keys = np.array([x[0] for x in session.query(ChanPair.corr_key).all()])

# len(cpairs)

# sta_ids = np.arange(130).astype(str)
sta_ids = np.array([sta.code for sta in stations])[::5]
stream = flow.get_continuous_fake(start_time, end_time, sta_ids, sr=dsr)
stream.sort()
data, sr, starttime = stream.as_array()

reload(xutil)
chan_names = np.array([f".{tr.stats.station}.{tr.stats.channel}" for tr in stream])
ckeys_all_str = [f"{k[0]}_{k[1]}"for k in xutil.unique_pairs(chan_names)]
ckeys = sorted(list(set(valid_corr_keys) & set(ckeys_all_str)))
ckeys_ix = xutil.index_ckeys_split(ckeys, chan_names)

dc = xchange.xcorr_ckeys_stack_slices(data, sr, ckeys_ix, cc_wlen_sec, keeplag_sec, stepsize_sec=stepsize_sec, whiten_freqs=whiten_freqs, onebit=onebit)

flow.fill_table_xcorrs(dc, ckeys, starttime.datetime, session, rhandle)

dists = flow.ckey_dists(ckeys, stations)
# plt.hist(dists.values(), bins=100)
############################################
xplot.im(dc)

# stations = sorted(stations, key=lambda x: x.code)
# [sta.code for sta in stations]
# import itertools
# list(itertools.product(['x', 'y'], ['X', 'Y', 'Z']))
