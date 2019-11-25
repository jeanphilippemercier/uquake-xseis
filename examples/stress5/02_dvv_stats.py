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

ckeys_all = session.query(XCorr.corr_key).all()
ckeys = np.unique(ckeys_all)
logger.info(f' {len(ckeys_all)} xcorr entries ({len(ckeys)} unique pairs)')


reload(xchange)

for i, ckey in enumerate(ckeys):
    # print(ckey)
    logger.info(f'Processing {i} / {len(ckeys)}')

    xcorrs = session.query(XCorr).filter_by(corr_key=ckey).order_by(XCorr.start_time.desc()).limit(nrecent).all()[::-1]
    flow.xcorr_load_waveforms(xcorrs, rhandle)

    logger.info(f'query returned {len(xcorrs)} xcorrs')

    dist = xcorrs[0].stationpair.dist
    coda_start_sec = dist / coda_start_vel
    print(f"{ckey}: {dist:.2f}m")

    for icc in range(1, len(xcorrs)):
        # print(icc)
        xref = xcorrs[icc - 1]
        xcur = xcorrs[icc]

        vals = xchange.dvv(xref.waveform, xcur.waveform, sr, dvv_wlen_sec, dvv_fband, coda_start_sec, coda_end_sec, dvv_outlier_clip=dvv_outlier_clip)
        xcur.dvv = [vals['dvv'], vals['coeff'], vals['n_outlier']]

    session.commit()

# plt.plot(xcur.waveform)
###############################################


ckeys_all = session.query(XCorr.corr_key).all()
ckeys = np.unique(ckeys_all)[:1000]
logger.info(f' {len(ckeys_all)} xcorr entries ({len(ckeys)} unique pairs)')


reload(xchange)
out = []

for i, ckey in enumerate(ckeys):
    # print(ckey)
    logger.info(f'Processing {i} / {len(ckeys)}')

    # ckey = ckeys[0]
    xcorrs = session.query(XCorr).filter_by(corr_key=ckey).order_by(XCorr.start_time.desc()).all()[::-1][1:]

    # stats = np.array([tuple(x.dvv) for x in xcorrs], dtype=[('dvv', float), ('coeff', float), ('n_outlier', float)])
    stats = np.array([x.dvv for x in xcorrs], dtype=np.float32)
    # stats = stats.view(np.recarray)
    # stats.ckey = ckey
    out.append(stats)

# out[0]
out = np.array(out)
out.shape
od = {'dvv': out[:, :, 0], 'coeff': out[:, :, 1], 'nout': out[:, :, 2]}
# dvv = out[:, :, 0]
# coeff = out[:, :, 1]
# nout = out[:, :, 2]
key = 'dvv'
key = 'coeff'
# key = 'nout'
xplot.im(od[key], norm=False)
plt.xlabel('time (20 min blocks)')
plt.ylabel('ccf #')
plt.title(key)
xplot.quicksave()

key = 'coeff'
od[key].shape
a = np.mean(od[key], axis=1)
plt.plot(a)

isort = np.argsort(a)[::-1]
a[isort]
plt.plot(a[isort])
ckeys[isort][:10]

xplot.im(od[key][isort], norm=False)
plt.xlabel('time (20 min blocks)')
plt.ylabel('ccf #')
plt.title(key)
xplot.quicksave()


key = 'coeff'
dat = od[key]


dchan = {}

for i, ck in enumerate(ckeys):
    k1, k2 = ck.split('_')
    if k1 not in dchan:
        dchan[k1] = []
    if k2 not in dchan:
        dchan[k2] = []
    dchan[k1].append(np.mean(dat[i]))
    dchan[k2].append(np.mean(dat[i]))

keys = []
vals = []
for k, v in dchan.items():
    keys.append(k[1:])
    vals.append(np.mean(v))

vals = np.array(vals)
# cn = keys[0]
chans = [session.query(Channel).filter_by(name=cn).first() for cn in keys]
locs = np.array([cn.station.location for cn in chans])
plt.plot(vals)

x, y, z = locs.T

fig = plt.figure(figsize=(12, 9), facecolor='white')
ax = fig.add_subplot(111, projection='3d')
x, y, z = locs.T
# ax.scatter(x, y, z, c='green', s=vals * 100)
im = ax.scatter(x, y, z, c=vals, s=50, alpha=0.5)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
plt.colorbar(im)
xplot.set_axes_equal(ax)
plt.tight_layout()

# plt.scatter(x, y, alpha=alpha, c=lvals, s=100, zorder=0)
plt.scatter(x, y, alpha=0.5, c=z, s=vals * 100, zorder=0)
plt.scatter(x, y, alpha=0.5, c=vals, s=50, zorder=0)

# else:
#     plt.scatter(x, y, alpha=alpha, s=6, zorder=0)
# x, y, z = locs[2900:3100].T
if lstep != 0:
    for i in range(0, locs.shape[0], lstep):
        plt.text(x[i], y[i], i)




plt.plot(dchan)

for stats in out[::5]:
    plt.plot(stats['dvv'], color='black', alpha=0.1)
    # plt.plot(stats['coeff'], color='black', alpha=0.1)


dvv = np.zeros(len(xcorrs), dtype={'names': ['dvv', 'coeff', 'n_outlier'], 'formats': ['f4', 'f4', 'f4']})

dvv = np.array([x.dvv for x in xcorrs], dtype={'names': ['dvv', 'coeff', 'n_outlier'], 'formats': ['f4', 'f4', 'f4']})

dvv = np.array([(1.0, 2), (3.0, 4)], dtype=[('x', float), ('y', int)])
vals = [tuple(x.dvv) for x in xcorrs]
dvv = np.array([tuple(x.dvv) for x in xcorrs], dtype=[('dvv', float), ('coeff', float),  ('n_outlier', float)])
dvv['coeff'].shape


# x = np.array([x.dvv for x in xcorrs[1:]]).astype={'names':['col1', 'col2'], 'formats':['i4','f4']})


##########################
key = 'coeff'
# key = 'n_outlier'
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



###############################

ckeys = np.unique(session.query(XCorr.corr_key).all())
logger.info(f'{len(ckeys)} unique xcorr pairs in db')

reload(xchange)
coda_start_vel = 3200.
coda_end_sec = 0.8
dvv_wlen_sec = 0.05
dvv_fband = np.array([60, 80, 250, 280])
# dvv_fband = np.array([60, 80, 320, 350])
dvv_outlier_clip = 0.1
# dvv_outlier_clip = None
sr = 1000.0

nrecent = 100

ckey = ckeys[5]
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

