from importlib import reload
import os
import numpy as np
from glob import glob
from datetime import datetime, timedelta

import redis
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xseis2 import xplot
from xseis2 import xutil
from xseis2 import xchange
from xseis2 import xchange_workflow as flow
from xseis2.xsql import Base, Channel, Station, StationPair, StationDvv, XCorr, ChanPair
# from sqlalchemy.sql import exists

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
whiten_freqs = np.array(params.whiten_corner_freqs)
cc_wlen_sec = params.wlen_sec
stepsize_sec = params.stepsize_sec
keeplag_sec = params.keeplag_sec
pair_dist_min = params.pair_dist_min
pair_dist_max = params.pair_dist_max
samplerate_decimated = params.samplerate_decimated
onebit = params.onebit_normalization
bad_chans = np.array(params.channel_blacklist)

# extra params #############################
onebit = False
nstack_min_percent = 50

noise_freqs = np.array([20, 30, 600, 650])
tt_change_percent = 0.001
noise_scale = 1.0

#######################################################
#  connect to databases
#######################################################

logger.info('Connect to psql database')
db_string = "postgres://postgres:postgres@localhost"
db = create_engine(db_string)
print(db.table_names())
session = sessionmaker(bind=db)()
Base.metadata.create_all(db)

##################################
logger.info('Connect to redis database')
rhandle = redis.Redis(host='localhost', port=6379, db=0)

#######################################################
#  build initial synthetic data
#######################################################

ckeys_db = np.array(session.query(ChanPair.name).all()).flatten()[::20]
logger.info(f'{len(ckeys_db)} potential corr keys')

sr = samplerate_decimated
keeplag = int(keeplag_sec * sr)

simdat = {}
for ck in ckeys_db:
    simdat[ck] = xutil.noise1d(keeplag, whiten_freqs, sr, scale=1)

#######################################################
#  xcorr and save to database
#######################################################

data_src = os.path.join(os.environ['SPP_COMMON'], "data_dump")
data_fles = np.sort(glob(os.path.join(data_src, '*.npz')))[:300]
# data_fles = np.sort(glob(os.path.join(data_src, '*.npz')))[200:300]
logger.info(f"nfiles: {len(data_fles)}")


for i, fname in enumerate(data_fles):
    logger.info(f"processing {i} / {len(data_fles)}")

    basename = os.path.basename(fname)

    data, sr, starttime, endtime, chan_names = flow.load_npz_continuous_meta(fname)

    ckeys_all_str = np.array([f"{k[0]}_{k[1]}"for k in xutil.unique_pairs(chan_names)])
    ckeys = np.array(sorted(list(set(ckeys_db) & set(ckeys_all_str))))
    ckeys_ix = xutil.index_ckeys_split(ckeys, chan_names)
    ckdict = dict(zip(ckeys, np.arange(len(ckeys))))
    logger.info(f'{len(ckeys)} matching ckeys')

    logger.info(f'sim fill')

    ncc = len(ckeys_ix)
    keeplag = int(keeplag_sec * sr)
    dc = np.zeros((ncc, keeplag * 2), dtype=np.float32)

    for k, v in ckdict.items():
        cc, half = xchange.stretch_and_mirror(simdat[k], sr, tt_change_percent, noise_freqs, noise_scale=noise_scale)
        dc[v] = cc
        simdat[k] = half

    # dc = xutil.add_noise(dc, whiten_freqs, sr, scale=1)
    nstack = np.ones(ncc) * 100

    logger.info(f'done')

    flow.fill_table_xcorrs(dc, sr, nstack, ckeys, starttime, endtime, session, rhandle, nstack_min_percent=nstack_min_percent)
