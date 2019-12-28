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

overwrite = False
# overwrite = True

################################
logger.info('Connect to psql database')
db_string = "postgres://postgres:postgres@localhost"
db = create_engine(db_string)
print(db.table_names())
session = sessionmaker(bind=db)()
if overwrite:
    flow.sql_drop_tables(db)
    session.commit()
Base.metadata.create_all(db)

##################################
logger.info('Connect to redis database')
rhandle = redis.Redis(host='localhost', port=6379, db=0)
if overwrite:
    rhandle.flushall()

#######################################################
#  xcorr processing params
#######################################################
# dsr = 2000.0
# PARAMS - config
# whiten_freqs = np.array([60, 80, 300, 320])
whiten_freqs = np.array([40, 50, 390, 400])
cc_wlen_sec = 10.0
stepsize_sec = cc_wlen_sec / 2
# stepsize_sec = cc_wlen_sec / 2
keeplag_sec = 1.0
# stacklen_sec = 100.0
min_pair_dist = 50
max_pair_dist = 800
nstack_min_percent = 50
onebit = False

#######################################################
#  create/fill sql databases
#######################################################

with open(os.path.join(os.environ['SPP_COMMON'], "stations.pickle"), 'rb') as f:
    stations_pkl = pickle.load(f)

if overwrite:
    # flow.fill_tables_sta_chan(stations_pkl, session)
    flow.fill_table_stations(stations_pkl, session)
    # stations = session.query(Station).all()
    flow.fill_table_station_pairs(session)
    flow.fill_table_chanpairs(session, min_pair_dist=min_pair_dist, max_pair_dist=max_pair_dist, bad_chans=xchange.ot_bad_chans())

# %%timeit
# cpairs = session.query(ChanPair).all()

# pairs = session.query(StationPair).filter(StationPair.dist.between(min_pair_dist, max_pair_dist)).filter().all()
# ckeys_db = flow.ckeys_from_stapairs(pairs)
# ckeys_db = xutil.ckeys_remove_chans(ckeys_db, xchange.ot_bad_chans())

ckeys_db = np.array(session.query(ChanPair.name).all()).flatten()[::2]
logger.info(f'{len(ckeys_db)} potential corr keys')

#######################################################
#  load files
#######################################################
reload(xchange)
reload(xutil)
reload(flow)
fproc = session.query(DataFile).all()
print(fproc)

data_src = os.path.join(os.environ['SPP_COMMON'], "data_dump")
# data_src = params['data_connector']['data_source']['location']
data_fles = np.sort(glob(os.path.join(data_src, '*.npz')))
data_fles = data_fles[:-1]
nfles = len(data_fles)
logger.info(f"nfiles: {len(data_fles)}, approx {nfles / 6 / 24:.2f} days")


#######################################################
#  xcorr and save to database
#######################################################

for i, fname in enumerate(data_fles):
    logger.info(f"processing {i} / {len(data_fles)}")

    basename = os.path.basename(fname)

    exists = session.query(DataFile.name).filter_by(name=basename).scalar() is not None

    if exists:
        print(f"already processed, skipping")
        continue

    data, sr, starttime, endtime, chan_names = flow.load_npz_continuous(fname)

    ckeys_all_str = np.array([f"{k[0]}_{k[1]}"for k in xutil.unique_pairs(chan_names)])
    ckeys = np.array(sorted(list(set(ckeys_db) & set(ckeys_all_str))))
    ckeys_ix = xutil.index_ckeys_split(ckeys, chan_names)
    logger.info(f'{len(ckeys)} matching ckeys')

    dc, nstack = xchange.xcorr_ckeys_stack_slices(data, sr, ckeys_ix, cc_wlen_sec, keeplag_sec, stepsize_sec=stepsize_sec, whiten_freqs=whiten_freqs, onebit=onebit)

    flow.fill_table_xcorrs(dc, sr, nstack, ckeys, starttime, endtime, session, rhandle, nstack_min_percent=nstack_min_percent)

    session.add(DataFile(name=basename, status=True))
    session.commit()

###################################################


# fname = data_fles[-2]
# data, sr, starttime, endtime, chan_names = flow.load_npz_continuous(fname)
# reload(flow)
# %%timeit
# flow.fill_table_xcorrs(dc, sr, nstack, ckeys, starttime, endtime, session, rhandle, nstack_min_percent=nstack_min_percent)

#####################
# # plt.plot(data[10])
# xplot.freq(data[10][:100000], sr)
# xplot.freq(dc[20], sr)
# xplot.im_freq(dc, sr)

