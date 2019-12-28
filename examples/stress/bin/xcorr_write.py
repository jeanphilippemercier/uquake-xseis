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

from pytz import utc
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
# import matplotlib.pyplot as plt

from microquake.core.settings import settings
from microquake.clients.ims import web_client

# plt.ion()

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
onebit = params.onebit_normalization
bad_chans = np.array(params.channel_blacklist)
req_length_sec = params.request_continuous_length_sec
req_lag_sec = params.request_continuous_lag_sec

sr_decimated = params.samplerate_decimated

#######################################################
#  connect to databases
#######################################################

logger.info('Connect to psql database')
db_string = "postgres://postgres:postgres@localhost"
db = create_engine(db_string)
print(db.table_names())
session = sessionmaker(bind=db)()
Base.metadata.create_all(db)

logger.info('Connect to redis database')
rhandle = redis.Redis(host='localhost', port=6379, db=0)

#######################################################
#  raw data request params
#######################################################
stations = settings.inventory.stations()

base_url = settings.get('ims_base_url')
api_base_url = settings.get('api_base_url')
inventory = settings.inventory
network_code = settings.NETWORK_CODE

# sites = [int(sta.code) for sta in stations]
sites = [18, 20, 22, 24, 26, 28, 30, 32, 41, 44, 46, 48, 50, 52, 54, 56, 58, 63, 65, 67, 69, 71, 79, 81, 90, 104, 106, 108, 110, 112, 115, 116, 117, 119, 121, 126, 140]
# sites = np.array(sites)[::5]
sites = np.array(sites)

req_lag = timedelta(seconds=-req_lag_sec)
req_length = timedelta(seconds=req_length_sec)

curtime = datetime.utcnow()
curtime = curtime.replace(tzinfo=utc, second=0, microsecond=0)
print(curtime)

req_start_time = UTCDateTime(curtime + req_lag)
req_end_time = UTCDateTime(req_start_time + req_length)
print(req_start_time)
print(req_end_time)

stream = web_client.get_continuous(base_url, req_start_time, req_end_time, sites, utc, network=network_code, format='binary')
stream.sort()
# stream0 = stream.copy()
# stream = stream0.copy()

nchan = len(stream)
longest_trace = np.max([len(tr.data) / tr.stats.sampling_rate for tr in stream])
nsamp = int(longest_trace * sr_decimated)
data = np.zeros((nchan, nsamp), dtype=np.float32)

chan_names = []

for i, tr in enumerate(stream):
    print(tr)
    sig = tr.data
    sr_raw = tr.stats.sampling_rate
    decf = int(sr_raw / sr_decimated)

    ixnan = np.where(np.isnan(sig))[0]
    sig -= np.nanmean(sig)
    sig = xutil.linear_detrend_nan(sig)
    sig[ixnan] = 0
    # sig = xutil.bandpass(sig, fband, sr)
    sig = xutil.filter(sig, 'lowpass', sr_decimated / 2 - 100, sr_decimated)
    sig = sig[::decf]

    i0 = int((tr.stats.starttime - req_start_time) * sr_decimated + 0.5)
    slen = min(len(sig), nsamp - i0)
    data[i, i0: i0 + slen] = sig[:slen]
    chan_names.append(f".{tr.stats.station}.{tr.stats.channel}")

chan_names = np.array(chan_names)

# tstring = req_start_time.__str__()
# fname = os.path.join(os.environ['SPP_COMMON'], "data_dump", f"ob_data_{tstring}")

# np.savez_compressed(fname, start_time=tstring, data=data, sr=sr_decimated, chans=chan_names)


#######################################################
#  xcorr
#######################################################
# data, sr, starttime, endtime, chan_names = flow.load_npz_continuous_meta(fname)
sr = sr_decimated
starttime = req_start_time.datetime
endtime = starttime + timedelta(seconds=data.shape[1] / sr)
nstack_min_percent = 20
onebit = False
reload(xchange)

ckeys_db = np.array(session.query(ChanPair.name).all()).flatten()[::2]
logger.info(f'{len(ckeys_db)} potential corr keys')


ckeys_all_str = np.array([f"{k[0]}_{k[1]}"for k in xutil.unique_pairs(chan_names)])
ckeys = np.array(sorted(list(set(ckeys_db) & set(ckeys_all_str))))
ckeys_ix = xutil.index_ckeys_split(ckeys, chan_names)
logger.info(f'{len(ckeys)} matching ckeys')

dc, nstack = xchange.xcorr_ckeys_stack_slices(data, sr, ckeys_ix, cc_wlen_sec, keeplag_sec, stepsize_sec=stepsize_sec, whiten_freqs=whiten_freqs, onebit=onebit)

flow.fill_table_xcorrs(dc, sr, nstack, ckeys, starttime, endtime, session, rhandle, nstack_min_percent=nstack_min_percent)


# for sig in data:
#     # pos_ratio = 100 * len(np.where(sig == 0)[0]) / len(sig)
#     # pos_ratio = 100 * len(np.where(sig > 0)[0]) / len(sig)
#     # print(pos_ratio)
#     print(len(np.where(sig == 0)[0]))


###################################################

