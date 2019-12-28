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
noise_scale = 0.1

# overwrite = False
overwrite = True

#######################################################
#  connect to databases
#######################################################

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
#  create/fill sql databases
#######################################################

with open(os.path.join(os.environ['SPP_COMMON'], "stations.pickle"), 'rb') as f:
    stations_pkl = pickle.load(f)

if overwrite:
    # flow.fill_tables_sta_chan(stations_pkl, session)
    flow.fill_table_stations(stations_pkl, session)
    # stations = session.query(Station).all()
    flow.fill_table_station_pairs(session)
    flow.fill_table_chanpairs(session, pair_dist_min=pair_dist_min, pair_dist_max=pair_dist_max, bad_chans=bad_chans)

#######################################################
#  load files
#######################################################

data_src = os.path.join(os.environ['SPP_COMMON'], "data_dump")
# data_src = params['data_connector']['data_source']['location']
data_fles = np.sort(glob(os.path.join(data_src, '*.npz')))

fproc = session.query(DataFile).all()
print(fproc)

logger.info(f"nfiles: {len(data_fles)}")

reload(xchange)
reload(xutil)
reload(flow)
# session.query(XCorr).delete()

#######################################################
#  xcorr and save to database
#######################################################

# %%timeit
# cpairs = session.query(ChanPair).all()

ckeys_db = np.array(session.query(ChanPair.name).all()).flatten()[::2]
logger.info(f'{len(ckeys_db)} potential corr keys')

sr = samplerate_decimated
keeplag = int(keeplag_sec * sr)

simdat = {}
for ck in ckeys_db:
    simdat[ck] = xutil.noise1d(keeplag, whiten_freqs, sr, scale=1)


# for i, fname in enumerate(data_fles[:-1]):
for i, fname in enumerate(data_fles[:100]):
    logger.info(f"processing {i} / {len(data_fles)}")

    basename = os.path.basename(fname)

    exists = session.query(DataFile.name).filter_by(name=basename).scalar() is not None

    if exists:
        print(f"already processed, skipping")
        continue

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

    # dc, nstack = xchange.xcorr_ckeys_stack_slices(data, sr, ckeys_ix, cc_wlen_sec, keeplag_sec, stepsize_sec=stepsize_sec, whiten_freqs=whiten_freqs, onebit=onebit)

    flow.fill_table_xcorrs(dc, sr, nstack, ckeys, starttime, endtime, session, rhandle, nstack_min_percent=nstack_min_percent)

    session.add(DataFile(name=basename, status=True))
    session.commit()

###################################################











# fname = data_fles[-2]
# data, sr, starttime, endtime, chan_names = flow.load_npz_continuous(fname)
reload(flow)
%%timeit
flow.fill_table_xcorrs(dc, sr, nstack, ckeys, starttime, endtime, session, rhandle, nstack_min_percent=nstack_min_percent)



# fwin = freq_window(freqs, wlen, sr)
nfreq = int(npts // 2 + 1)
# fwin = whiten_window(freqs, nfreq, sr)
fwin = freq_window(freqs, padlen, sr)
nfreq = len(fwin)

w = xutil.freq_window(whiten_freqs, wlen, sr)
plt.plot(w)

reload(xutil)
reload(xchange)
sig = xutil.noise1d(wlen, whiten_freqs, sr, scale=1)
# xplot.freq(sig, sr)

s1 = xchange.stretch_and_mirror(sig, sr, tt_change_percent, noise_freqs, noise_scale=0.2)
plt.plot(s1)
plt.plot(sig[::-1])

ncc = len(ckeys_ix)
keeplag = int(keeplag_sec * sr)
dc = np.zeros((ncc, keeplag * 2), dtype=np.float32)
dc = xutil.add_noise(dc, whiten_freqs, sr, scale=1)
nstack = np.ones(ncc) * 100



#####################
# # plt.plot(data[10])
# xplot.freq(data[10][:100000], sr)
xplot.freq(dc[20], sr)
xplot.im_freq(dc, sr)


# padlen = int(cc_wlen_sec * sr * 2)
# nfreq = int(padlen // 2 + 1)

# freqs = np.fft.rfftfreq(padlen, 1.0 / sr)

# whiten_freqs = np.array(whiten_freqs)
# w0 = xutil.freq_window(whiten_freqs, padlen, sr)
# w1 = xutil.whiten_window(whiten_freqs, nfreq, sr)

# plt.plot(freqs, w0)
# plt.plot(freqs, w1)


# def whiten_sig(sig, sr, whiten_freqs):
#     whiten_win = xutil.freq_window(whiten_freqs, len(sig), sr)
#     fsig = np.fft.rfft(sig)
#     fsig = whiten_win * xutil.phase(fsig)
#     return np.fft.irfft(fsig)


# whiten_freqs = np.array([40, 50, 380, 400])

# sig = data[0]
# # sig = whiten_sig(sig, sr, whiten_freqs)
# whiten_win = xutil.freq_window(whiten_freqs, len(sig), sr)
# fsig = np.fft.rfft(sig)
# fsig = whiten_win * xutil.phase(fsig)
# sig = np.fft.irfft(fsig)
# xplot.freq(sig, sr)







###################################
# plt.plot(nstack)

# nchan = len(data)
# stack_flag = np.ones(nchan, dtype=bool)
# stack_flag[0] = False
# stack_flag[92] = False
# stack_flag[ckeys_ix]

# ikeep = np.where(np.sum(stack_flag[ckeys_ix], axis=1) == 2)[0]


############################################

###########################

# wild = f"%115%"
# xcorrs = session.query(XCorr).filter(XCorr.stationpair_name.like(wild)).all()
# cc = xcorrs[0]

# wild = f".115.%.116.%"
wild = f".115.Z%116.Z"
xcorrs = session.query(XCorr).filter(XCorr.corr_key.like(wild)).all()
# xcorrs = session.query(XCorr).all()
print(len(xcorrs))
xc = xcorrs[0]

reload(flow)
flow.xcorr_load_waveforms(xcorrs, rhandle)

dd = [x.stationpair.dist for x in xcorrs]
dat = np.array([x.waveform for x in xcorrs])

im = xplot.im(dat, norm=True)

reload(xplot)
im = xplot.im(dat[np.argsort(dd)], norm=False)
xplot.HookImageMax(im)
#######################

sig1 = np.mean(dat[:4], axis=0)
sig2 = np.mean(dat[17:], axis=0)
dist = xcorrs[0].stationpair.dist

# keeplag_sec = 1.0

reload(xchange)
coda_start_vel = 3200.
coda_end_sec = 0.8
dvv_wlen_sec = 0.05

coda_start_sec = dist / coda_start_vel

fband_sig = np.array([60, 80, 320, 350])
# fband_sig = np.array([50, 80, 170, 180])
whiten_freqs = fband_sig

dvv_outlier_clip = 0.1
# dvv_outlier_clip = None

reload(xutil)
reload(xchange)
vals = xchange.dvv(sig1, sig2, sr, dvv_wlen_sec, whiten_freqs, coda_start_sec, coda_end_sec, dvv_outlier_clip=dvv_outlier_clip)

reload(xchange)
xchange.plot_dvv(vals)
xv = xutil.xcorr_lagtimes(len(sig1))
plt.plot(xv, xutil.maxnorm(sig1, 5), alpha=0.3)
plt.plot(xv, xutil.maxnorm(sig2, 5), alpha=0.3)

xplot.quicksave()


########################


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






# import psycopg2
# conn = psycopg2.connect("dbname=postgres user=postgres host='localhost' password='postgres'")
# conn = psycopg2.connect("dbname=postgres user=postgres")
# conn = psycopg2.connect("dbname='template1' user='dbuser' host='localhost' password='dbpass'")
# conn.close()










###########################
# wild = f".115.%.116.%"
# xcorrs = session.query(XCorr).filter(XCorr.corr_key.like(wild)).all()

# wild = f"%115%"
# xcorrs = session.query(XCorr).filter(XCorr.stationpair_name.like(wild)).all()
# cc = xcorrs[0]

xcorrs = session.query(XCorr).all()

reload(flow)
flow.xcorr_load_waveforms(xcorrs, rhandle)

dd = [x.stationpair.dist for x in xcorrs]
dat = np.array([x.waveform for x in xcorrs])


reload(xplot)
im = xplot.im(dat[np.argsort(dd)], norm=False)
xplot.HookImageMax(im)

mx = np.max(dc, axis=1)
plt.plot(mx)
imax = np.argmax(mx)
plt.plot(dat[imax])
xcorrs[imax]
# dat
# ckeys[0:10]
wild = f".115.%.116.%"
xcorrs = session.query(XCorr).filter(XCorr.corr_key.like(wild)).all()

reload(flow)
flow.xcorr_load_waveforms(xcorrs, rhandle)

dat = np.array([x.waveform for x in xcorrs])

plt.plot(dat.T)


ccfs = [xutil.bytes_to_array(rhandle.get(cc.waveform_redis_key)) for cc in out]
ccfs = np.array(ccfs)

im = xplot.im(ccfs, norm=False)
xplot.HookImageMax(im)


a = out[0]








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


# ckey = ckeys[0]
# k1, k2 = ckey.split("_")
# spair = f"{k1.split('.')[1]}_{k2.split('.')[1]}"
# c1, c2 = k1[1:], k2[1:]

# spair = [k[1:] for k in ckey.split("_")]

# name = f"{sta1.name}_{sta2.name}"


# xcorrs = session.query(XCorr).filter(XCorr.channel1.station.name =="115").all()
# xcorrs = session.query(XCorr).filter(XCorr.channel1_name == "115.X").all()


# ckey = ckeys[0]
# c1, c2 = [k[1:] for k in ckey.split("_")]

# dkey = f"{str(starttime)} {ckey}"

# XCorr(start_time=starttime, corr_key=ckey, waveform_redis_key=dkey)

# pipe.execute()  # add data to redis
# session.add_all(rows)  # add metadata rows to sql
# session.commit()
