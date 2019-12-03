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
from xseis2.xsql import VelChange, XCorr, ChanPair, Base

from loguru import logger
from obspy import UTCDateTime
# from microquake.core.settings import settings
# from microquake.core.stream import Trace, Stream
# from pytz import utc
from microquake.plugin.site.core import read_csv
import matplotlib.pyplot as plt
import pickle

plt.ion()

# filename = "/home/phil/data/oyu/spp_common/sensors.csv"
# inv = read_csv(filename, site_code='', has_header=True)
# stations = inv.stations()
# stations = sorted(stations, key=lambda x: x.code)

with open(os.path.join(os.environ['SPP_COMMON'], "stations.pickle"), 'rb') as f:
    stations = pickle.load(f)


################################
logger.info('Connect to psql database')

db_string = "postgres://postgres:postgres@localhost"
db = create_engine(db_string)
print(db.table_names())
session = sessionmaker(bind=db)()
flow.sql_drop_tables(db)
session.commit()
# session.close()

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
stacklen_sec = 100.0
onebit = False
min_pair_dist = 100
max_pair_dist = 1000

####################################

###############################

reload(flow)
flow.fill_table_chanpairs(stations, session, min_pair_dist, max_pair_dist)
# db_corr_keys = np.array([x[0] for x in session.query(ChanPair.corr_key).filter(ChanPair.inter_dist < max_pair_dist).filter(ChanPair.inter_dist > min_pair_dist).all()])
db_corr_keys = np.array([x[0] for x in session.query(ChanPair.corr_key).all()])

logger.info(f'{len(db_corr_keys)} potential corr keys')

###############################

data_src = os.path.join(os.environ['SPP_COMMON'], "data_dump")
# data_src = params['data_connector']['data_source']['location']
data_fles = np.sort(glob(os.path.join(data_src, '*.npz')))

fname = data_fles[0]
# fh = np.load(fname, start_time=tstring, data=data, sr=dsr, chans=chan_names)

# npz = np.load(fname)
with np.load(fname) as npz:
    sr = float(npz['sr'])
    chan_names = npz['chans']
    starttime = UTCDateTime(str(npz['start_time']))
    data = npz['data'].astype(np.float32)
    data[data == 0] = -1

ratios = np.array([100 * len(np.where(sig > 0)[0]) / len(sig) for sig in data])

# plt.plot(ratios)

rawdat = data
nchan, nsamp = rawdat.shape

cclen = int(cc_wlen_sec * sr)
slices = xutil.build_slice_inds(0, nsamp, cclen)

out = np.zeros((nchan, len(slices)))
for i, sl in enumerate(slices):
    print(f"stacking slice {i} / {len(slices)}")

    for isig in range(nchan):
        sig = rawdat[isig, sl[0]:sl[1]].copy()
        pos_ratio = 100 * len(np.where(sig > 0)[0]) / len(sig)
        out[isig][i] = pos_ratio

reload(xplot)
xplot.im(out, norm=False)
plt.xlabel("slice #")
plt.ylabel("channel #")
plt.title("positive value ratio")

xplot.quicksave()

plt.plot(data[84])
plt.plot(out.T, marker='o', alpha=0.2)
xplot.quicksave()


ix = np.argmin(ratios)
ix = 7
sig = data[ix]
# plt.plot(sig)
# sig = np.zeros(len(sig))
pos_ratio = 100 * len(np.where(sig > 0)[0]) / len(sig)

nsamp = len(sig)
cclen = int(cc_wlen_sec * sr)
slices = xutil.build_slice_inds(0, nsamp, cclen)

ratios = []
for i, sl in enumerate(slices):
    print(f"stacking slice {i} / {len(slices)}")
    win = sig[sl[0]:sl[1]]
    pos_ratio = 100 * len(np.where(win > 0)[0]) / len(win)
    ratios.append(pos_ratio)

plt.plot(ratios, marker='o')
################################################



#################################
# sta_ids = np.arange(130).astype(str)
# sta_ids = np.array([sta.code for sta in stations])[::10]
# stream = flow.get_continuous_fake(start_time, end_time, sta_ids, sr=dsr)
# stream.sort()
# reload(xchange)

# data, sr, starttime, chan_names = xchange.stream_decompose(stream)
# np.allclose(data[10], stream[10].data)
# data[30] = np.roll(data[10], 100)
# data[[30, 10], :100000] = 0
# print(chan_names[30], chan_names[10])


fname = data_fles[0]
# npz = np.load(fname)
with np.load(fname) as npz:
    sr = float(npz['sr'])
    chan_names = npz['chans']
    starttime = UTCDateTime(str(npz['start_time']))
    data = npz['data'].astype(np.float32)
    data[data == 0] = -1


reload(xutil)
ckeys_all_str = [f"{k[0]}_{k[1]}"for k in xutil.unique_pairs(chan_names)]
ckeys = sorted(list(set(db_corr_keys) & set(ckeys_all_str)))
ckeys_ix = xutil.index_ckeys_split(ckeys, chan_names)
logger.info(f'{len(ckeys)} matching ckeys')

dc = xchange.xcorr_ckeys_stack_slices(data, sr, ckeys_ix, cc_wlen_sec, keeplag_sec, stepsize_sec=stepsize_sec, whiten_freqs=whiten_freqs, onebit=onebit)

# session.query(XCorr).delete()
flow.fill_table_xcorrs(dc, ckeys, starttime.datetime, session, rhandle)

################################################


flow.fill_table_chanpairs(stations, session, min_pair_dist, max_pair_dist)
db_corr_keys = np.array([x[0] for x in session.query(ChanPair.corr_key).all()])
logger.info(f'{len(db_corr_keys)} potential corr keys')


reload(xchange)
reload(flow)
curtime = datetime.utcnow()
req_times = [curtime + timedelta(seconds=i * stacklen_sec) for i in range(5)]
session.query(XCorr).delete()

for i, req_start_time in enumerate(req_times):
    req_end_time = req_start_time + timedelta(seconds=stacklen_sec)

    sta_ids = np.array([sta.code for sta in stations])[::10]
    stream = flow.get_continuous_fake(req_start_time, req_end_time, sta_ids, sr=dsr)
    stream.sort()
    data, sr, starttime, chan_names = xchange.stream_decompose(stream)

    reload(xutil)
    ckeys_all_str = [f"{k[0]}_{k[1]}"for k in xutil.unique_pairs(chan_names)]
    ckeys = sorted(list(set(db_corr_keys) & set(ckeys_all_str)))
    ckeys_ix = xutil.index_ckeys_split(ckeys, chan_names)
    logger.info(f'{len(ckeys)} matching ckeys')

    dc = xchange.xcorr_ckeys_stack_slices(data, sr, ckeys_ix, cc_wlen_sec, keeplag_sec, stepsize_sec=stepsize_sec, whiten_freqs=whiten_freqs, onebit=onebit)

    flow.fill_table_xcorrs(dc, ckeys, starttime.datetime, session, rhandle)

############################################


# def measure_dvv_xcorrs(session, rhandle):

coda_start_vel = 3200.
coda_end_sec = 0.8
wlen_sec = 0.05
# whiten_freqs = np.array([80, 100, 250, 300])
nrecent = 10

ckeys = np.unique(session.query(XCorr.corr_key).all())
logger.info(f'{len(ckeys)} unique xcorr pairs in db')

chan_pairs = session.query(ChanPair).filter(ChanPair.corr_key.in_(ckeys)).all()

cpair = chan_pairs[0]
dist = cpair.inter_dist
ckey = cpair.corr_key

coda_start_sec = dist / coda_start_vel
print(f"{ckey}: {dist:.2f}m")

ccfs = session.query(XCorr).filter_by(corr_key=ckey).order_by(XCorr.start_time.desc()).limit(nrecent).all()[::-1]

for icc in range(1, len(ccfs)):
    cc_ref = ccfs[icc - 1]
    cc_cur = ccfs[icc]
    sig_ref = xutil.bytes_to_array(rhandle.get(cc_ref.waveform_redis_key))
    sig_cur = xutil.bytes_to_array(rhandle.get(cc_cur.waveform_redis_key))
    dvv, error = xchange.dvv(sig_ref, sig_cur, sr, wlen_sec, whiten_freqs, coda_start_sec, coda_end_sec)
    cc_cur.delta_velocity = dvv
    cc_cur.error = error


plt.plot(sig_ref)
plt.plot(sig_cur)








for cpair in chan_pairs:
    dist = cpair.inter_dist
    ckey = cpair.corr_key

    coda_start_sec = dist / coda_start_vel
    print(f"{ckey}: {dist:.2f}m")

    ccfs = session.query(XCorr).filter_by(ckey=ckey).order_by(XCorr.time.desc()).limit(nrecent).all()[::-1]

    for icc in range(1, len(ccfs)):
        # print(i)
        cc_ref = ccfs[icc - 1]
        cc_cur = ccfs[icc]
        sig_ref = xutil.bytes_to_array(rhandle.get(cc_ref.data))
        sig_cur = xutil.bytes_to_array(rhandle.get(cc_cur.data))
        dvv, error = xchange.dvv(sig_ref, sig_cur, sr, wlen_sec, whiten_freqs, coda_start_sec, coda_end_sec)
        cc_cur.dvv = dvv
        cc_cur.error = error


# xplot.im(dc, norm=False)
# xplot.im(dc, norm=False)
















# stations = sorted(stations, key=lambda x: x.code)
# [sta.code for sta in stations]
# import itertools
# list(itertools.product(['x', 'y'], ['X', 'Y', 'Z']))


# data, sr, starttime, chan_names = xchange.stream_decompose(stream)
# cclen = int(cc_wlen_sec * sr)
# nsamp = data.shape[1]
# slices = xutil.build_slice_inds(0, nsamp, cclen)
# sl = slices[0]
# dat = data[:, sl[0]:sl[1]].copy()
# dat[[0, 20]] = 0

# random_amp = 1e-9
# ix_bad = np.where(np.sum(np.abs(dat), axis=1) == 0)[0]
# dat[ix_bad[0]] = np.random.uniform(-random_amp, random_amp, dat.shape[1])
# dat[ix_bad[1]] = np.random.uniform(-random_amp, random_amp, dat.shape[1])

# xplot.im(dat, norm=False)
# xplot.im(dat, norm=True)

# plt.plot(dat[0])