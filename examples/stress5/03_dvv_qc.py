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
# import pickle

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


###########################

reload(xutil)
reload(xchange)
vel_s = 3200
coda_start_vel = 0.7 * vel_s
coda_end_sec = 0.4
dvv_wlen_sec = 0.05
dvv_fband = np.array([60, 70, 230, 250])
dvv_outlier_clip = 0.1
# dvv_outlier_clip = None
sr = 1000.0
wlengths = coda_start_vel / dvv_fband
nrecent = 200
nrow_avg = 15
# wild = f"%120.%.126%"
wild = f"%118%"
# ck_all = session.query(XCorr.corr_key).all()
# ckeys = np.unique(session.query(XCorr.corr_key).all())[::100]
ckeys = session.query(XCorr.corr_key).filter(XCorr.corr_key.like(wild)).all()
ckeys = np.unique(ckeys)
logger.info(f'{len(ckeys)} unique xcorr pairs in db')

stats = []
dists = []

for ckey in ckeys:
    # print(ckey)

    xcorrs = session.query(XCorr).filter_by(corr_key=ckey).order_by(XCorr.start_time.desc()).limit(nrecent).all()[::-1]
    flow.xcorr_load_waveforms(xcorrs, rhandle)

    logger.info(f'query returned {len(xcorrs)} xcorrs')

    xcorrs_dat = np.array([x.waveform for x in xcorrs])
    xcorrs_dat = xutil.average_adjacent_rows(xcorrs_dat, nrow_avg=nrow_avg)

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


key = 'dvv'
# key = 'n_outlier'
# key = 'coeff'
vals = np.array([pair[key] for pair in stats])
vals[vals == 0] = np.nan
# plt.plot(vals.T, alpha=0.2, marker='o')
xv = np.linspace(0, vals.shape[1] * nrow_avg / 3, vals.shape[1])
[plt.scatter(xv, y, alpha=0.1, color='black') for y in vals]
plt.plot(xv, np.nanmean(vals, axis=0))
plt.plot(xv, np.nanmedian(vals, axis=0))
plt.ylabel(key)
plt.xlabel("time (hours)")
xplot.quicksave()
# cumsum #########################

key = 'dvv'
vals = np.array([pair[key] for pair in stats])
vals = np.cumsum(vals, axis=1)
# vals[vals == 0] = np.nan
# plt.plot(vals.T, alpha=0.2, marker='o')
xv = np.linspace(0, vals.shape[1] * nrow_avg / 3, vals.shape[1])
[plt.scatter(xv, y, alpha=0.1, color='black') for y in vals]
plt.plot(xv, np.nanmean(vals, axis=0))
plt.plot(xv, np.nanmedian(vals, axis=0))
plt.ylabel(key)
plt.xlabel("time (hours)")
xplot.quicksave()


#############################

# vals = np.array([pair[key] for pair in stats])
# vals[vals == 0] = np.nan
# xplot.im(vals, norm=False)

# plt.hist(vals.flatten(), bins=10)
# plt.hist(vals[:, 0])
# plt.hist(vals[:, 1])

############################################

# key = 'coeff'
# key = 'n_outlier'
# vals = np.array([np.mean(p[key]) for p in stats])
# imax = np.argmin(vals)
vals = np.array([np.mean(p['coeff']) for p in stats])
imax = np.argmax(vals)
ckey = ckeys[imax]
nrow_avg = 12

xcorrs = session.query(XCorr).filter_by(corr_key=ckey).order_by(XCorr.start_time.desc()).limit(nrecent).all()[::-1]
flow.xcorr_load_waveforms(xcorrs, rhandle)
xcorrs_dat_raw = np.array([x.waveform for x in xcorrs])
xcorrs_dat = xutil.average_adjacent_rows(xcorrs_dat_raw, nrow_avg=nrow_avg)

dist = xcorrs[0].stationpair.dist
coda_start_sec = dist / coda_start_vel
print(f"{ckey}: {dist:.2f}m")

out = []
for icc in range(1, len(xcorrs_dat)):
    xref_sig = xcorrs_dat[icc - 1]
    xcur_sig = xcorrs_dat[icc]

    dvv_out = xchange.dvv(xref_sig, xcur_sig, sr, dvv_wlen_sec, dvv_fband, coda_start_sec, coda_end_sec, dvv_outlier_clip=dvv_outlier_clip)
    out.append(dvv_out)

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

################################

# ix = np.argmin(vals['coeff'])
ix = np.argmax(stat['coeff'])
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


stations = session.query(Station).filter(Station.name != '134').all()
locs = np.array([x.location for x in stations])
lbls = np.array([x.name for x in stations])
# plt.plot(vals)
x, y, z = locs.T

fig = plt.figure(figsize=(12, 9), facecolor='white')
ax = fig.add_subplot(111, projection='3d')
x, y, z = locs.T
# ax.scatter(x, y, z, c='green', s=vals * 100)
im = ax.scatter(x, y, z, c='green', s=50, alpha=0.5)
for i, lbl in enumerate(lbls):
    ax.text(x[i], y[i], z[i], lbl)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
plt.colorbar(im)
xplot.set_axes_equal(ax)
plt.tight_layout()


###################################
