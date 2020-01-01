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
nstack_min_percent = 30
onebit = False
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
#  xcorr and save to database
#######################################################

def sta_ckeys(stations):

    ckeys = []
    for sta in stations:
        out = session.query(ChanPair.name).filter(ChanPair.name.like(f"%.{sta}.%")).all()
        ckeys.extend([x[0] for x in out])
    return np.array(ckeys).flatten()


# sta = 112
# ckeys_db = session.query(ChanPair.name).filter(ChanPair.name.like(f"%.{sta}.%")).all()
# ckeys_db = np.array(ckeys_db).flatten()

ckeys_db = sta_ckeys([112, 116, 140])

# ckeys_db = np.array(session.query(ChanPair.name).all()).flatten()[:]
logger.info(f'{len(ckeys_db)} potential corr keys')


data_src = os.path.join(os.environ['SPP_COMMON'], "data_dump")
data_fles = np.sort(glob(os.path.join(data_src, '*.npz')))[:]
logger.info(f"nfiles: {len(data_fles)}")


for i, fname in enumerate(data_fles[868:]):
    logger.info(f"processing {i} / {len(data_fles)}")

    basename = os.path.basename(fname)
    data, sr, starttime, endtime, chan_names = flow.load_npz_continuous(fname)

    ckeys_all_str = np.array([f"{k[0]}_{k[1]}"for k in xutil.unique_pairs(chan_names)])
    ckeys = np.array(sorted(list(set(ckeys_db) & set(ckeys_all_str))))
    ckeys_ix = xutil.index_ckeys_split(ckeys, chan_names)
    logger.info(f'{len(ckeys_ix)} matching ckeys')

    if len(ckeys_ix) == 0:
        continue

    dc, nstack = xchange.xcorr_ckeys_stack_slices(data, sr, ckeys_ix, cc_wlen_sec, keeplag_sec, stepsize_sec=stepsize_sec, whiten_freqs=whiten_freqs, onebit=onebit)

    flow.fill_table_xcorrs(dc, sr, nstack, ckeys, starttime, endtime, session, rhandle, nstack_min_percent=nstack_min_percent)
