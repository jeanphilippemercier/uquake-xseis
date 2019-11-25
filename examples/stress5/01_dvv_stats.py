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
from xseis2.xsql import Base, Channel, Station, StationPair, VelChange, XCorr, ChanPair

import itertools

from loguru import logger
from obspy import UTCDateTime
# from pytz import utc
import matplotlib.pyplot as plt
import pickle

plt.ion()

################################
logger.info('Connect to psql database')
db_string = "postgres://postgres:postgres@localhost"
db = create_engine(db_string)
print(db.table_names())
session = sessionmaker(bind=db)()
# flow.sql_drop_tables(db)
# session.commit()
Base.metadata.create_all(db)
# session.commit()

##################################
logger.info('Connect to redis database')
rhandle = redis.Redis(host='localhost', port=6379, db=0)
# rhandle.flushall()
###################################

ckeys_all = session.query(XCorr.corr_key).all()
ckeys = np.unique(ckeys_all)
logger.info(f' {len(ckeys_all)} xcorr entries ({len(ckeys)} unique pairs)')

reload(xchange)
coda_start_vel = 3400.
coda_end_sec = 0.8
dvv_wlen_sec = 0.05
dvv_fband = np.array([60, 80, 250, 280])
# dvv_fband = np.array([60, 80, 320, 350])
dvv_outlier_clip = 0.1
# dvv_outlier_clip = None
sr = 1000.0

nrecent = 100

ckey = ckeys[50]
ckey = ".115.Z_.116.Z"
xcorrs = session.query(XCorr).filter_by(corr_key=ckey).order_by(XCorr.start_time.desc()).limit(nrecent).all()[::-1]
flow.xcorr_load_waveforms(xcorrs, rhandle)

logger.info(f'query returned {len(xcorrs)} xcorrs')

dist = xcorrs[0].stationpair.dist
coda_start_sec = dist / coda_start_vel
print(f"{ckey}: {dist:.2f}m")

out = []

for icc in range(1, len(xcorrs)):
    print(icc)
    xref = xcorrs[icc - 1]
    xcur = xcorrs[icc]

    reload(xutil)
    reload(xchange)
    vals = xchange.dvv(xref.waveform, xcur.waveform, sr, dvv_wlen_sec, dvv_fband, coda_start_sec, coda_end_sec, dvv_outlier_clip=dvv_outlier_clip)
    out.append(vals)

reload(xutil)
vals = xutil.to_dict_of_lists(out)
print(list(vals.keys()))
plt.plot(vals['dvv'])
plt.plot(vals['n_outlier'])
plt.plot(vals['coeff'])

# ix = np.argmin(vals['coeff'])
ix = np.argmax(vals['coeff'])
# plt.plot(out)
reload(xchange)
xchange.plot_dvv(out[ix])

sig1 = xcorrs[ix].waveform
sig2 = xcorrs[ix + 1].waveform

xv = xutil.xcorr_lagtimes(len(sig1))
plt.plot(xv, xutil.maxnorm(sig1, 5), alpha=0.3)
plt.plot(xv, xutil.maxnorm(sig2, 5), alpha=0.3)

xutil.pearson_coeff(sig1, sig2)

dat = np.array([x.waveform for x in xcorrs])
# im = xplot.im(dat, norm=True)
# reload(xplot)
im = xplot.im(dat, norm=False)
xplot.HookImageMax(im)

xplot.im_freq(dat, sr)
#####################################################

###########################

reload(xutil)
reload(xchange)
coda_start_vel = 3500.
coda_end_sec = 0.7
dvv_wlen_sec = 0.05
dvv_fband = np.array([60, 80, 250, 280])
dvv_outlier_clip = 0.1
# dvv_outlier_clip = None
sr = 1000.0
wlengths = coda_start_vel / dvv_fband

nrecent = 200

ckeys = np.unique(session.query(XCorr.corr_key).all())[::100]
logger.info(f'{len(ckeys)} unique xcorr pairs in db')

stats = []
dists = []

for ckey in ckeys:
    # print(ckey)

    xcorrs = session.query(XCorr).filter_by(corr_key=ckey).order_by(XCorr.start_time.desc()).limit(nrecent).all()[::-1]
    flow.xcorr_load_waveforms(xcorrs, rhandle)

    logger.info(f'query returned {len(xcorrs)} xcorrs')

    xcorrs_dat = np.array([x.waveform for x in xcorrs])
    xcorrs_dat = xutil.average_adjacent_rows(xcorrs_dat, nrow_avg=12)

    dist = xcorrs[0].stationpair.dist
    dists.append(dist)

    coda_start_sec = dist / coda_start_vel
    print(f"{ckey}: {dist:.2f}m")

    out = []

    for icc in range(1, len(xcorrs_dat)):
        xref_sig = xcorrs_dat[icc - 1]
        xcur_sig = xcorrs_dat[icc]

        vals = xchange.dvv(xref_sig, xcur_sig, sr, dvv_wlen_sec, dvv_fband, coda_start_sec, coda_end_sec, dvv_outlier_clip=dvv_outlier_clip)
        out.append(vals)

    stats.append(xutil.to_dict_of_lists(out))

print(list(stats[0].keys()))


# key = 'coeff'
key = 'n_outlier'
# key = 'dvv'
for pair in stats:
    plt.plot(pair[key], alpha=0.2, marker='o')
plt.ylabel(key)
plt.xlabel("time (60 min intervals)")
xplot.quicksave()


key1 = 'coeff'
# key2 = 'dvv'
# key2 = 'coeff'
key2 = 'n_outlier'
for pair in stats:
    plt.scatter(pair[key1], pair[key2], alpha=0.5)
plt.xlabel(key1)
plt.ylabel(key2)
xplot.quicksave()

key = 'coeff'
vals = [np.mean(p[key]) for p in stats]
plt.scatter(dists, vals, alpha=0.5)
plt.ylabel(key)
plt.xlabel("dists")
xplot.quicksave()

plt.plot(vals)

key1 = 'coh'
vals1 = [np.mean(np.mean(p[key])) for p in stats]
key2 = 'coeff'
vals2 = [np.mean(p[key]) for p in stats]
plt.scatter(vals1, vals2, alpha=0.5)
plt.xlabel(key1)
plt.ylabel(key2)
xplot.quicksave()
########################

# key = 'coeff'
key = 'n_outlier'
vals = np.array([np.mean(p[key]) for p in stats])
# imax = np.argmax(vals)
imax = np.argmin(vals)
ckey = ckeys[imax]

xcorrs = session.query(XCorr).filter_by(corr_key=ckey).order_by(XCorr.start_time.desc()).limit(nrecent).all()[::-1]
flow.xcorr_load_waveforms(xcorrs, rhandle)
xcorrs_dat_raw = np.array([x.waveform for x in xcorrs])
xcorrs_dat = xutil.average_adjacent_rows(xcorrs_dat_raw, nrow_avg=12)

dist = xcorrs[0].stationpair.dist
coda_start_sec = dist / coda_start_vel
print(f"{ckey}: {dist:.2f}m")

out = []
for icc in range(1, len(xcorrs_dat)):
    xref_sig = xcorrs_dat[icc - 1]
    xcur_sig = xcorrs_dat[icc]

    vals = xchange.dvv(xref_sig, xcur_sig, sr, dvv_wlen_sec, dvv_fband, coda_start_sec, coda_end_sec, dvv_outlier_clip=dvv_outlier_clip)
    out.append(vals)

stat = xutil.to_dict_of_lists(out)


reload(xutil)

xcorrs_dat.shape
xplot.im(xcorrs_dat, norm=False)

mx = np.max(xcorrs_dat, axis=1) * 100
plt.plot(mx[1:])
plt.plot(stat['coeff'])
plt.plot(stat['dvv'])
plt.plot(stat['n_outlier'])

# plt.plot(mx)


###################################

# dat = np.array([x.waveform for x in xcorrs])
# out = xutil.average_adjacent_rows(dat, nrow_avg=3)
# xplot.im(out)


###########################

reload(xutil)
reload(xchange)
coda_start_vel = 3500.
coda_end_sec = 0.8
dvv_wlen_sec = 0.05
dvv_fband = np.array([60, 80, 250, 280])
dvv_outlier_clip = 0.1
# dvv_outlier_clip = None
sr = 1000.0
wlengths = coda_start_vel / dvv_fband

nrecent = 100

ckeys = np.unique(session.query(XCorr.corr_key).all())[::100]
logger.info(f'{len(ckeys)} unique xcorr pairs in db')

stats = []

for ckey in ckeys:
    # print(ckey)

    xcorrs = session.query(XCorr).filter_by(corr_key=ckey).order_by(XCorr.start_time.desc()).limit(nrecent).all()[::-1]
    flow.xcorr_load_waveforms(xcorrs, rhandle)

    logger.info(f'query returned {len(xcorrs)} xcorrs')

    dist = xcorrs[0].stationpair.dist
    coda_start_sec = dist / coda_start_vel
    print(f"{ckey}: {dist:.2f}m")

    out = []

    for icc in range(1, len(xcorrs)):
        # print(icc)
        xref = xcorrs[icc - 1]
        xcur = xcorrs[icc]

        vals = xchange.dvv(xref.waveform, xcur.waveform, sr, dvv_wlen_sec, dvv_fband, coda_start_sec, coda_end_sec, dvv_outlier_clip=dvv_outlier_clip)
        out.append(vals)

    stats.append(xutil.to_dict_of_lists(out))


key = 'coeff'
key = 'n_outlier'
# key = 'dvv'
for pair in stats:
    plt.plot(pair[key], alpha=0.5, marker='o')
plt.ylabel(key)
plt.xlabel("time (20 min intervals)")
xplot.quicksave()
###################################

