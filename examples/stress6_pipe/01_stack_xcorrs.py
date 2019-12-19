from importlib import reload
import os
import numpy as np
from glob import glob
from datetime import datetime, timedelta
import itertools
import time

import redis
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xseis2 import xplot
from xseis2 import xutil
from xseis2 import xchange
from xseis2 import xchange_workflow as flow
from xseis2.xsql import Base, Station, XCorr, ChanPair, StationDvv
# from xseis2 import xsql

from loguru import logger
import matplotlib.pyplot as plt

from microquake.core.settings import settings

plt.ion()

#######################################################
#  xcorr processing params
#######################################################

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
#  combine multiple
#######################################################

nhour_stack = settings.COMPUTE_XCORRS.stack_length_hours
nrecent = 9999

ckeys_db = np.array(session.query(XCorr.corr_key).distinct().order_by(XCorr.corr_key.asc()).all()).flatten()
logger.info(f'{len(ckeys_db)} unique xcorr pairs in db')
reload(flow)

# delete already stacked corrs
delete_q = XCorr.__table__.delete().where(XCorr.length == nhour_stack)
session.execute(delete_q)
session.commit()


for i, ckey in enumerate(ckeys_db):
    logger.info(f"processing {i} / {len(ckeys_db)}")

    xcorrs = session.query(XCorr).filter_by(corr_key=ckey).order_by(XCorr.start_time.desc()).limit(nrecent).all()[::-1]
    flow.xcorr_load_waveforms(xcorrs, rhandle)

    xcorrs_stack = flow.time_group_and_stack_xcorrs(xcorrs, nhour_stack)
    flow.write_xcorrs(xcorrs_stack, session, rhandle)


#######################################################
#  measure dvv
#######################################################

ckeys_db = np.array(session.query(XCorr.corr_key).distinct().order_by(XCorr.corr_key.asc()).all()).flatten()
# ckeys_db = np.unique(session.query(XCorr.corr_key).all())
logger.info(f'{len(ckeys_db)} unique xcorr pairs in db')

# params = settings.COMPUTE_VELCHANGE
# whiten_freqs = np.array(params.whiten_corner_freqs)
# cc_wlen_sec = params.wlen_sec
# stepsize_sec = params.stepsize_sec
# keeplag_sec = params.keeplag_sec
# pair_dist_min = params.pair_dist_min
# pair_dist_max = params.pair_dist_max
# samplerate_decimated = params.samplerate_decimated
# onebit = params.onebit_normalization


reload(xutil)
reload(xchange)
vel_s = 3200
coda_start_vel = 0.7 * vel_s
coda_end_sec = 0.5
dvv_wlen_sec = 0.02
dvv_freq_lims = np.array([70, 240])
# dvv_fband = np.array([60, 80, 240, 250])
wlen_relative = dvv_freq_lims / dvv_wlen_sec
nwin_welch = 10
step_factor = 10


for i, ckey in enumerate(ckeys_db):
    logger.info(f"processing {i} / {len(ckeys_db)}")

    cpair = session.query(ChanPair).filter_by(name=ckey).first()
    if cpair is None:
        print(f"{ckey} missing")
        continue

    xcorrs = session.query(XCorr).filter(XCorr.length == nhour_stack).filter_by(corr_key=ckey).order_by(XCorr.start_time.desc()).limit(nrecent).all()[::-1]
    # xcorrs = session.query(XCorr).filter_by(corr_key=ckey).filter(XCorr.length == nhour_stack).order_by(XCorr.start_time.desc()).limit(nrecent).all()[::-1]
    flow.xcorr_load_waveforms(xcorrs, rhandle)

    xcorrs_dat = np.array([x.waveform for x in xcorrs])
    ncc, cclen = xcorrs_dat.shape

    dist = cpair.dist
    sr = xcorrs[0].samplerate
    coda_start_sec = dist / coda_start_vel
    print(f"{ckey}: {dist:.2f}m")

    for icc in range(1, len(xcorrs_dat)):
        xref_sig = xcorrs_dat[icc - 1]
        xcur_sig = xcorrs_dat[icc]

        # dvv_out = xchange.dvv(xref_sig, xcur_sig, sr, dvv_wlen_sec, dvv_fband, coda_start_sec, coda_end_sec, dvv_outlier_clip=dvv_outlier_clip)
        # dvv_out = xchange.dvv_phase_no_weights(xref_sig, xcur_sig, sr, dvv_wlen_sec, dvv_freq_lims, coda_start_sec, coda_end_sec, step_factor=step_factor)
        dvv_out = xchange.dvv_phase(xref_sig, xcur_sig, sr, dvv_wlen_sec, dvv_freq_lims, coda_start_sec, coda_end_sec, nwin_welch=nwin_welch, step_factor=step_factor)
        xcorrs[icc].velocity_change = dvv_out['dvv']
        xcorrs[icc].error = dvv_out['regress'][2]
        xcorrs[icc].pearson = dvv_out['coeff']

    session.commit()

reload(xchange)
plt.plot(xref_sig)
plt.plot(xcur_sig)

%%timeit
# xcorrs = session.query(XCorr).filter(XCorr.length == nhour_stack).filter_by(corr_key=ckey).order_by(XCorr.start_time.desc()).limit(nrecent).all()[::-1]
xcorrs = session.query(XCorr).filter_by(corr_key=ckey).filter(XCorr.length == nhour_stack).order_by(XCorr.start_time.desc()).limit(nrecent).all()[::-1]
#######################################################
#  average dvv stations
#######################################################

reload(flow)
flow.measure_dvv_stations(session, nhour_stack)

# ckeys = flow.unique_ordered(session, XCorr.corr_key)
# stas = flow.unique_ordered(session, Station.name)

stas = flow.unique_ordered(session, StationDvv.station)

vals = []
for sta in stas:
    out = session.query(StationDvv).filter(StationDvv.station == sta).order_by(StationDvv.start_time.asc()).all()
    dvv = [x.velocity_change for x in out]
    vals.append(dvv)
    # plt.plot(dvv, alpha=0.4)
    plt.plot(np.cumsum(dvv))

#################################
ckeys = flow.unique_ordered(session, XCorr.corr_key)

for i, ckey in enumerate(ckeys):
    logger.info(f"processing {i} / {len(ckeys)}")

    cpair = session.query(ChanPair).filter_by(name=ckey).first()
    if cpair is None:
        print(f"{ckey} missing")
        continue

    out = session.query(XCorr).filter_by(corr_key=ckey).filter(XCorr.velocity_change.isnot(None)).order_by(XCorr.start_time.asc()).all()
    dvv = [x.velocity_change for x in out]
    # vals.append(dvv)
    # plt.plot(dvv)
    plt.plot(dvv, alpha=0.4)
    # plt.plot(np.cumsum(dvv), alpha=0.2)


#######################################################
#  read dvv
#######################################################
xcorrs = session.query(XCorr).filter(XCorr.velocity_change.isnot(None)).all()
xcorrs = session.query(XCorr).filter(XCorr.error.isnot(None)).all()

err = [x.error for x in xcorrs]
err = [x.coeff for x in xcorrs]
err = [x.velocity_change for x in xcorrs]
plt.hist(err, bins=100)

v1 = [x.error for x in xcorrs]
v2 = [x.velocity_change for x in xcorrs]

plt.scatter(v1, v2, alpha=0.1, s=3)

#######################################################
#  benchmark
#######################################################

 # query is bottleneck 17ms
%%timeit
xcorrs = session.query(XCorr).all()
xcorrs = session.query(XCorr).filter_by(corr_key=ckey).filter(XCorr.length == nhour_stack).order_by(XCorr.start_time.desc()).limit(nrecent).all()[::-1]
%%timeit
flow.xcorr_load_waveforms(xcorrs, rhandle)

ckey = ckeys_db[0]
%%timeit
xcorrs = session.query(XCorr).filter_by(corr_key=ckey).first()


%%timeit
xcorrs_dat = np.array([x.waveform for x in xcorrs])
ncc, cclen = xcorrs_dat.shape

dist = xcorrs[0].stationpair.dist
sr = xcorrs[0].samplerate
coda_start_sec = dist / coda_start_vel
print(f"{ckey}: {dist:.2f}m")



#######################################################
#  test one
#######################################################


# ckeys_db = np.unique(session.query(XCorr.corr_key).all())
# ckeys_db = session.query(XCorr.corr_key).distinct().all()
%%timeit
ckeys_db = np.array(session.query(XCorr.corr_key).distinct().order_by(XCorr.corr_key.asc()).all()).flatten()
%%timeit
ckeys_db2 = np.unique(session.query(XCorr.corr_key).all())

(ckeys_db == ckeys_db2).all()
(np.sort(ckeys_db) == np.sort(ckeys_db2)).all()


ckeys = xchange.ot_best_pairs()

# ckey = '.131.X_.79.X'
ckey = ckeys[1]
nrow_avg = 10
reload(xchange)

xcorrs = session.query(XCorr).filter_by(corr_key=ckey).order_by(XCorr.start_time.desc()).limit(nrecent).all()[::-1]
flow.xcorr_load_waveforms(xcorrs, rhandle)
xcorrs_dat_raw = np.array([x.waveform for x in xcorrs])
xcorrs_dat = xutil.average_adjacent_rows(xcorrs_dat_raw, nrow_avg=nrow_avg)


xcorrs = session.query(XCorr).filter_by(corr_key=ckey).filter(XCorr.length < 4).order_by(XCorr.start_time.desc()).limit(nrecent).all()[::-1]
x = xcorrs[0]
len(xcorrs)


reload(xutil)
reload(flow)
nhour_stack = 5

xgroups, dbins = flow.group_xcorrs_by_time(xcorrs, nhour_stack)
[len(x) for x in xgroups]

xcorrs_stack = []
for xg, new_start_time in zip(xgroups, dbins):
    xnew = flow.stack_xcorrs(xg, new_start_time, nhour_stack)
    xcorrs_stack.append(xnew)

flow.write_xcorrs(xcorrs_stack, session, rhandle)

flow.xcorr_load_waveforms(xcorrs_stack, rhandle)

xcorrs_stack
sr = xcorrs_stack[0].samplerate
sigs = np.array([x.waveform for x in xcorrs_stack])
xplot.im(sigs, norm=False)
xplot.im_freq(sigs, sr, norm=False)
xplot.im(xcorrs_dat_raw, norm=False)
# xplot.im_freq(xcorrs_dat_raw, sr, norm=True)

xg = xgroups[-8]
new_nstack = np.mean(np.array([x.nstack / 100 * x.length for x in xg]))
nhour_data = np.sum(np.array([x.nstack / 100 * x.length for x in xg]))
new_nstack = nhour_data / new_length * 100

ig = 4
xg = xgroups[ig]
new_start_time = dbins[ig]
new_length = nhour_stack
xnew = stack_xcorrs(xg, new_start_time, new_length)


sigs = np.array([x.waveform for x in xg])

plt.plot(xnew.waveform)
plt.plot(sigs[0])
plt.plot(sigs[1])


wlen = timedelta(hours=nhour_stack)

dates = np.array([x.start_time for x in xcorrs])
start = xutil.hour_round_down(np.min(dates))
stop = xutil.hour_round_up(np.max(dates))
dbins = xutil.datetime_bins(start, stop, wlen)

xgroups = [[] for i in range(len(dbins))]

for xc in xcorrs:
    ind = np.argmin(np.abs(dbins - xc.start_time))
    xgroups[ind].append(xc)

xgroups = np.array(xgroups)
ikeep = np.where(np.array([len(x) for x in xgroups]) > 0)[0]
xgroups[ikeep]
[len(x) for x in xgroups[ikeep]]
# dbins = [start + timedelta(hours=nhour_stack) for i in range(5)]


dbins - start
np.min(dbins - dates[5])
np.argmin(np.abs(dbins - dates[5]))

# xgroups = np.array(len(dbins), dtype=object)


dbins = xutil.datetime_bins(start, stop, wlen)
dbins = np.arange(start, stop, wlen)


dbins - stop
dates - stop


##############################

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


# ckeys_db = np.array(session.query(ChanPair.name).all()).flatten()
# ckeys_db = np.unique(session.query(XCorr.corr_key).all())
# ckeys = np.unique(session.query(XCorr.corr_key).all())[::100]
# ckeys = session.query(XCorr.corr_key).filter(XCorr.corr_key.like(wild)).all()
