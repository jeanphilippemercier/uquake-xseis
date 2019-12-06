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


ckeys_db = np.unique(session.query(XCorr.corr_key).all())
# ckeys = np.unique(session.query(XCorr.corr_key).all())[::100]
# ckeys = session.query(XCorr.corr_key).filter(XCorr.corr_key.like(wild)).all()
logger.info(f'{len(ckeys_db)} unique xcorr pairs in db')


#################################

reload(xutil)
reload(xchange)
vel_s = 3200
# vel_s = 8000
coda_start_vel = 0.7 * vel_s
coda_end_sec = 0.5
dvv_wlen_sec = 0.02
dvv_freq_lims = np.array([70, 240])
# dvv_fband = np.array([60, 80, 240, 250])
wlen_relative = dvv_freq_lims / dvv_wlen_sec
nwin_welch = 10
step_factor = 10

sr = 2000.0
dvv_outlier_clip = 0.1
nrecent = 200

################################
ckeys = xchange.ot_best_pairs()

# ckey = '.131.X_.79.X'
ckey = ckeys[1]
nrow_avg = 10
reload(xchange)


xcorrs = session.query(XCorr).filter_by(corr_key=ckey).order_by(XCorr.start_time.desc()).limit(nrecent).all()[::-1]
flow.xcorr_load_waveforms(xcorrs, rhandle)
xcorrs_dat_raw = np.array([x.waveform for x in xcorrs])
xcorrs_dat = xutil.average_adjacent_rows(xcorrs_dat_raw, nrow_avg=nrow_avg)

kept = np.array([x.nstack for x in xcorrs])
dtimes = np.array([x.start_time for x in xcorrs])
dist = xcorrs[0].stationpair.dist
coda_start_sec = dist / coda_start_vel
print(f"{ckey}: {dist:.2f}m")

ncc, cclen = xcorrs_dat.shape
times = xutil.xcorr_lagtimes(cclen, sr)
# icoda = np.where((times < coda_end_sec) & (times > coda_start_sec))
icoda = np.where((np.abs(times) < coda_end_sec) & (np.abs(times) > coda_start_sec))
out = []

for icc in range(1, len(xcorrs_dat)):
    xref_sig = xcorrs_dat[icc - 1]
    xcur_sig = xcorrs_dat[icc]

    # pcc.append(xutil.pearson_coeff(xref_sig[icoda], xcur_sig[icoda]))

    # dvv_out = xchange.dvv(xref_sig, xcur_sig, sr, dvv_wlen_sec, dvv_fband, coda_start_sec, coda_end_sec, dvv_outlier_clip=dvv_outlier_clip)
    # dvv_out = xchange.dvv_phase_no_weights(xref_sig, xcur_sig, sr, dvv_wlen_sec, dvv_freq_lims, coda_start_sec, coda_end_sec, step_factor=step_factor)
    dvv_out = xchange.dvv_phase(xref_sig, xcur_sig, sr, dvv_wlen_sec, dvv_freq_lims, coda_start_sec, coda_end_sec, nwin_welch=nwin_welch, step_factor=step_factor)
    out.append(dvv_out)

stat = xutil.to_dict_of_lists(out)

reload(xutil)
reload(xplot)
#####################
plt.plot(stat['dvv'])
# plt.plot(pcc)
ix = np.argmax(stat['coeff'])
ix = -1

vals = out[ix]
reload(xchange)
xchange.plot_dvv(vals)


freqs = vals['freqs']
xvals = vals['xvals']
cohs = vals['cohs']
# cohs[cohs < 0.1] = 0.1
xplot.im(cohs ** 2, norm=False, extent=[freqs[0], freqs[-1], xvals[0], xvals[-1]])
weights = 1.0 / (1.0 / (cohs ** 2) - 1.0)
# weights = 1.0 / (1.0 / (cohs) - 1.0)
# weights = 1.0 / (cohs)
# weights = 1.0 / (cohs ** 2) - 1.0
im = xplot.im(weights, norm=False, extent=[freqs[0], freqs[-1], xvals[0], xvals[-1]])
xplot.HookImageMax(im)



##########################
ncc, cclen = xcorrs_dat.shape
times = xutil.xcorr_lagtimes(cclen, sr)

im = xplot.im(xcorrs_dat, times=times)
# im = xplot.im(xcorrs_dat, norm=False, times=times)
xplot.axvline(coda_start_sec)
xplot.axvline(coda_end_sec)
xplot.HookImageMax(im)


xplot.im_freq(xcorrs_dat, sr, norm=False)
vel = dist / (np.argmax(xcorrs_dat, axis=1) - cclen // 2) * sr
plt.plot(vel)
xplot.freq(xcorrs_dat[0], sr)

mx = np.max(xcorrs_dat, axis=1) * 100
plt.plot(mx[1:])
plt.plot(stat['coeff'])
plt.plot(pcc)
plt.plot(stat['dvv'])
plt.plot(stat['n_outlier'])

########################

# ix = np.argmin(vals['coeff'])
ix = np.argmax(stat['coeff'])
ix = np.argmax(stat['dvv'])
ix = np.argmax(pcc)
# plt.plot(out)
ix = 0
reload(xchange)
xchange.plot_dvv(out[ix])

sig1 = xcorrs[ix].waveform
sig2 = xcorrs[ix + 1].waveform

xv = xutil.xcorr_lagtimes(len(sig1), sr)
plt.plot(xv, xutil.maxnorm(sig1))
plt.plot(xv, xutil.maxnorm(sig2))
xplot.axvline(coda_start_sec)
xplot.axvline(coda_start_sec + dvv_wlen_sec)
xplot.axvline(coda_end_sec)
plt.plot(out[ix]['xvals'] / sr, out[ix]['coh'])

cohs = out[ix]['cohs']
freqs = out[ix]['freqs']
xplot.im(cohs, norm=False, times=freqs)

plt.plot(freqs, np.mean(cohs, axis=0))

xutil.pearson_coeff(sig1, sig2)

dat = np.array([x.waveform for x in xcorrs])
# im = xplot.im(dat, norm=True)
# reload(xplot)
im = xplot.im(dat, norm=False)
xplot.HookImageMax(im)

xplot.im_freq(dat, sr)
#####################################################


fname = os.path.join(os.environ['SPP_COMMON'], "xcorrs_sample.npz")
np.savez(fname, data=xcorrs_dat, sr=sr, ckey=ckey, dist=dist)
with np.load(fname) as npz:
    sr = float(npz['sr'])
    data = npz['data']
    dist = float(npz['dist'])
    ckey = str(npz['ckey'])



plt.plot(xref_sig)
xw = xref_sig.copy()
xw[icoda] = np.nan
plt.plot(xw)



s1 = xutil.split_causals(xref_sig)
s2 = xutil.split_causals(xcur_sig)
plt.plot(s1[0])
plt.plot(s2[0])
plt.plot(s1[1])
plt.plot(s2[1])
# plt.plot(mx)

mx = np.max(xcorrs_dat, axis=1) * 100
plt.plot(mx)
plt.plot(kept / 100)


stats = []
dists = []
nkept = []
ckeys = []

for ckey in ckeys_db:
    # print(ckey)
    xcorrs = session.query(XCorr).filter_by(corr_key=ckey).order_by(XCorr.start_time.desc()).limit(nrecent).all()[::-1]
    if len(xcorrs) < 100:
        continue

    flow.xcorr_load_waveforms(xcorrs, rhandle)

    logger.info(f'query returned {len(xcorrs)} xcorrs')

    nkept.append(np.array([x.nstack for x in xcorrs]))

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
        pc = xutil.pearson_coeff(xref_sig, xcur_sig)
        out.append({'coeff': pc, 'mx': np.max(xcur_sig)})

        # vals = xchange.dvv(xref_sig, xcur_sig, sr, dvv_wlen_sec, dvv_fband, coda_start_sec, coda_end_sec, dvv_outlier_clip=dvv_outlier_clip)
        # vals = xchange.dvv_phase(xref_sig, xcur_sig, sr, dvv_wlen_sec, dvv_fband[[0, -1]], coda_start_sec, coda_end_sec, step_factor=4)

        # out.append(vals)

    stats.append(xutil.to_dict_of_lists(out))
    ckeys.append(ckey)


print(list(stats[0].keys()))

ckeys = np.array(ckeys)
###################################
vals1 = np.array([np.nanmean(p['coeff']) for p in stats])
# vals1 = np.array([np.mean(p['mx']) for p in stats])
vals2 = np.array([np.nanmean(v) for v in nkept])
# vals2 = np.array([np.mean(p['coeff']) for p in stats])
col = plt.scatter(vals1, vals2, alpha=0.2)
reload(xplot)
hook = xplot.HookLasso(col)
ckeys[hook.ind]
######
ik = hook.ind
ik = np.arange(len(ckeys))
# k, v = xutil.ckey_to_chan_stats(ckeys[ik], vals1[ik])
k, v = xutil.ckey_to_chan_stats(ckeys[ik], np.ones(len(ik)))
vm = np.array([np.sum(x) for x in v])
isort = np.argsort(vm)[::-1]
plt.plot(vm[isort], marker='o')
plt.xticks(np.arange(len(k)), k[isort])
k[isort]

#####################


vals = np.array([len(p['coeff']) for p in stats])
vals = np.array([p['coeff'] for p in stats])
xplot.im(vals, norm=False)

vals = np.array([np.mean(p['mx']) for p in stats])
vals = np.array([np.mean(p['coeff']) for p in stats])
plt.scatter(vals, dists, alpha=0.2)
plt.plot(vals)
plt.hist(vals, bins=20)
############################

reload(xchange)
sites = xchange.ot_keep_sites().astype(str)
# stations = [session.query(Station).filter(Station.name == str(sta)).first() for sta in sites]
stations = session.query(Station).filter(Station.name != '134').all()
locs = np.array([x.location for x in stations])
lbls = np.array([x.name for x in stations])
# plt.plot(vals)

# vals = np.array([np.mean(p['mx']) for p in stats])
vals = np.array([np.mean(p['coeff']) for p in stats])
plt.plot(vals, marker='x', alpha=0.2)
ikeep = np.argsort(vals)[::-1][::10]
# plt.plot(np.sort(vals)[::-1])
# ikeep = np.argsort(vals)[::-1][-200::2]
# ikeep = np.argsort(vals)[::-1][:100]
# ikeep = np.where(vals < 0.01)[0]
# ikeep = np.where(vals < 0.2)[0]
ikeep = np.where(vals > 0.6)[0]
# ikeep = np.where(vals < 0.02)[0]
# ikeep = hook.ind
print(len(ikeep))
ckeep = ckeys[ikeep]
vkeep = vals[ikeep]

reload(xplot)
fig = plt.figure(figsize=(12, 9), facecolor='white')
ax = fig.add_subplot(111, projection='3d')
x, y, z = locs.T
# ax.scatter(x, y, z, c='green', s=vals * 100)
im = ax.scatter(x, y, z, c='green', s=50, alpha=0.5)
for i, lbl in enumerate(lbls):
    if lbl in sites:
        ax.text(x[i], y[i], z[i], lbl)

# clrs = xplot.v2color(vkeep)
clrs, cmap = xplot.build_cmap(vkeep)
alpha = 1.0

# for i, ck in enumerate(ckeep):
for i, ck in enumerate(ckeep):
    xcorrs = session.query(XCorr).filter_by(corr_key=ck).first()
    l1 = xcorrs.stationpair.station1.location
    l2 = xcorrs.stationpair.station2.location
    x, y, z = np.array([l1, l2]).T
    ax.plot(x, y, z, alpha=alpha, color=clrs[i], linewidth=2)

cmap._A = []
cb = plt.colorbar(cmap)
# plt.colorbar(im)

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
xplot.set_axes_equal(ax)
plt.tight_layout()
plt.title('good pairs (coeff)')
xplot.quicksave()

###################################


# key = 'coeff'
# key = 'n_outlier'
# vals = np.array([np.mean(p[key]) for p in stats])
# imax = np.argmin(vals)
vals = np.array([np.mean(p['mx']) for p in stats])
imax = np.argmin(vals)
# imax = np.argmax(vals)
ckey = ckeys[imax]
ckey = '.131.X_.79.X'
nrow_avg = 1

xcorrs = session.query(XCorr).filter_by(corr_key=ckey).order_by(XCorr.start_time.desc()).limit(nrecent).all()[::-1]
flow.xcorr_load_waveforms(xcorrs, rhandle)
xcorrs_dat_raw = np.array([x.waveform for x in xcorrs])
xcorrs_dat = xutil.average_adjacent_rows(xcorrs_dat_raw, nrow_avg=nrow_avg)

kept = np.array([x.nstack for x in xcorrs])

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

ncc, cclen = xcorrs_dat.shape
xplot.im(xcorrs_dat, norm=False)
xplot.im_freq(xcorrs_dat, sr, norm=False)
vel = dist / (np.argmax(xcorrs_dat, axis=1) - cclen // 2) * sr
plt.plot(vel)
xplot.freq(xcorrs_dat[0], sr)

mx = np.max(xcorrs_dat, axis=1) * 100
plt.plot(mx[1:])
plt.plot(stat['coeff'])
plt.plot(stat['dvv'])
plt.plot(stat['n_outlier'])

# plt.plot(mx)


mx = np.max(xcorrs_dat, axis=1) * 100
plt.plot(mx)
plt.plot(kept / 100)
###########################################


def ckey_to_chan_stats(ckeys, vals):

    d = dict()
    for i, ck in enumerate(ckeys):
        c1, c2 = ck.split('_')
        if c1 not in d:
            d[c1] = [vals[i]]
            continue
        if c2 not in d:
            d[c2] = [vals[i]]
            continue

        d[c1].append(vals[i])
        d[c2].append(vals[i])

    return np.array(list(d.keys())), np.array(list(d.values()))


# vals = np.array([np.nanmean(p['mx']) for p in stats])
vals = np.array([np.nanmean(p['coeff']) for p in stats])
k, v = ckey_to_chan_stats(ckeys, vals)
vm = np.array([np.mean(x) for x in v])
isort = np.argsort(vm)
plt.plot(vm[isort])
plt.xticks(np.arange(len(k)), k[isort])
k[isort]

# vals = np.array([np.nanmean(p['mx']) for p in stats])
vals = np.array([np.nanmean(v) for v in kept])
k, v = ckey_to_chan_stats(ckeys, vals)
vm = np.array([np.mean(x) for x in v])
isort = np.argsort(vm)
plt.plot(vm[isort])
plt.xticks(np.arange(len(k)), k[isort])





####################################

# key = 'dvv'
# key = 'n_outlier'
key = 'coeff'
key = 'mx'
vals = np.array([pair[key] for pair in stats])
# vals[vals == 0] = np.nan
# plt.plot(vals.T, alpha=0.2, marker='o')
# xv = np.linspace(0, vals.shape[1] * nrow_avg / 3, vals.shape[1])
# xv = np.linspace(0, vals.shape[1] * nrow_avg / 3, vals.shape[1])
[plt.scatter(np.arange(len(y)), y, alpha=0.1, color='black') for y in vals]
# plt.plot(xv, np.nanmean(vals, axis=0))
# plt.plot(xv, np.nanmedian(vals, axis=0))
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
vals = np.array([np.nanmean(p['coeff']) for p in stats])
plt.plot(xutil.maxnorm(vals), label='coeff')
vals = np.array([np.mean(p['mx']) for p in stats])
plt.plot(xutil.maxnorm(vals), label='mx')
plt.legend()

# vals = np.array([pair[key] for pair in stats])
# vals[vals == 0] = np.nan
# xplot.im(vals, norm=False)

# plt.hist(vals.flatten(), bins=10)
# plt.hist(vals[:, 0])
# plt.hist(vals[:, 1])

############################################


###################################

stations = session.query(Station).all()


xlims = [651000, 652000]
ylims = [4766500, 4768000]

keep = []
for sta in stations:
    x, y, z = sta.location
    if x > xlims[0] and x < xlims[1] and y > ylims[0] and y < ylims[1]:
        keep.append(sta)

stations = keep[::2]
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

np.sort(np.array([sta.name for sta in stations], dtype=int))
