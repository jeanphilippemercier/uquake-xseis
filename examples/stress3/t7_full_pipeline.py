import os
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from importlib import reload

import time
from xseis2 import xutil
from xseis2 import xchange
from xseis2 import xplot
from xseis2 import xio
from spp.core.settings import settings
from glob import glob

from xseis2.h5stream import H5Stream

from sqlalchemy import MetaData
from sqlalchemy.sql import select
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sqlalchemy import create_engine
from xseis2.xsql import VelChange, XCorr, ChanPair, Base

import redis
plt.ion()

##################################
ddir = os.path.join(os.environ['SPP_COMMON'], 'ot_cont')
hfname = os.path.join(ddir, '10hr_1000hz.h5')
# hfname = os.path.join(ddir, '10hr_sim.h5')
hstream = H5Stream(hfname)


################################
db_string = "postgres://postgres:postgres@localhost"
db = create_engine(db_string)
print(db.table_names())
# Session = sessionmaker(bind=db)
# session = Session()
# session.close()
session = sessionmaker(bind=db)()

metadata = MetaData(db)
metadata.reflect()

for table in reversed(metadata.sorted_tables):
    table.drop(db)
    session.commit()

Base.metadata.create_all(db)
session.commit()


##################################
rhandle = redis.Redis(host='localhost', port=6379, db=0)
rhandle.flushall()
###################################

# x = settings.inventory.stations()[0]
# c = x.channels[0]

reload(xutil)
chans = []

for sta in settings.inventory.stations():
    # loc = sta.loc
    for chan in sta.channels:
        chans.append(f"{sta.code}.{chan.code}")
chans = np.array(chans)

ckeys = xutil.unique_pairs(chans)
ckeys = xutil.ckeys_remove_intersta_str(ckeys)
ckeys = np.array([f"{ck[0]}_{ck[1]}" for ck in ckeys])


sites = [station.code for station in settings.inventory.stations()]
site_locs = [station.loc for station in settings.inventory.stations()]
ldict = dict(zip(sites, site_locs))

dists = dict()
for ck in ckeys:
    c1, c2 = ck.split('_')
    dd = xutil.dist(ldict[c1[:-2]], ldict[c2[:-2]])
    dists[ck] = dd

rows = []
for k, v in dists.items():
    rows.append(ChanPair(ckey=k, dist=v))
    # print(k, v)

session.add_all(rows)  # add rows to sql
session.commit()

# ckeys = session.query(ChanPair.ckey).all()
# dists = xchange.ckey_dists(ckeys)
# dist = session.query(ChanPair.dist).filter(ChanPair.ckey == ck).first()[0]
# %timeit dist = session.query(ChanPair.dist).filter(ChanPair.ckey == ck).first()[0]


##################################

# channels to correlate can either be (1) taken from db (2) or params
chans = hstream.channels
sr_raw = hstream.samplerate
sr = sr_raw

# PARAMS - config
whiten_freqs = np.array([60, 80, 320, 350])
cclen = 20.0
keeplag = 1.0
stepsize = cclen
onebit = True
stacklen = 3600.0
# stacklen = 1000

# Params to function
times = [hstream.starttime + timedelta(seconds=i * stacklen) for i in range(5)]

for t0 in times:
    print(t0)
    t1 = t0 + timedelta(seconds=stacklen)

    # python generator which yields slices of data
    datgen = hstream.slice_gen(t0, t1, chans, cclen, stepsize=stepsize)

    dc, ckeys_ix = xchange.xcorr_stack_slices(
        datgen, chans, cclen, sr_raw, sr, keeplag, whiten_freqs, onebit=onebit)
    ckeys = [f"{ck[0]}_{ck[1]}" for ck in chans[ckeys_ix]]

    pipe = rhandle.pipeline()
    rows = []
    for i, sig in enumerate(dc):
        # print(i)
        ckey = ckeys[i]
        dkey = f"{str(t0)} {ckey}"
        pipe.set(dkey, xio.array_to_bytes(sig))
        rows.append(XCorr(time=t0, ckey=ckey, data=dkey))

    pipe.execute()  # add data to redis
    session.add_all(rows)  # add rows to sql
    session.commit()

###################################################################

# sr = self.params.cc_samplerate
# wlen_sec = self.params.wlen_sec
# coda_start_vel = self.params.coda_start_velocity
# coda_end_sec = self.params.coda_end_sec
# whiten_freqs = self.params.whiten_corner_freqs


coda_start_vel = 3200.
sr = 1000.0
vel = 3200.
# dist = dists[ckey]
# dist = 300
coda_end_sec = 0.8
wlen_sec = 0.05
whiten_freqs = np.array([80, 100, 250, 300])
nrecent = 10
reload(xchange)
########################################

#############################################################################

ckeys = np.unique(session.query(XCorr.ckey).all())
# dist = session.query(ChanPair.dist).filter(ChanPair.ckey.in_(ckeys)).all()
cpairs = session.query(ChanPair).filter(ChanPair.ckey.in_(ckeys)).filter(ChanPair.dist < 1000).all()

# dists = xchange.ckey_dists(ckeys)
# for i, ckey in enumerate(ckeys):
# for ckey in ckeys[:]:

for cpair in cpairs:

    dist = cpair.dist
    ckey = cpair.ckey
    # dist = session.query(ChanPair.dist).filter(ChanPair.ckey == ckey).first()[0]
    coda_start_sec = dist / coda_start_vel
    print(f"{ckey}: {dist:.2f}m")

    ccfs = session.query(XCorr).filter_by(ckey=ckey).order_by(XCorr.time.desc()).limit(nrecent).all()[::-1]

    for icc in range(1, len(ccfs)):
        # print(i)
        cc_ref = ccfs[icc - 1]
        cc_cur = ccfs[icc]
        sig_ref = xio.bytes_to_array(rhandle.get(cc_ref.data))
        sig_cur = xio.bytes_to_array(rhandle.get(cc_cur.data))
        dvv, error = xchange.dvv(sig_ref, sig_cur, sr, wlen_sec, whiten_freqs, coda_start_sec, coda_end_sec)
        cc_cur.dvv = dvv
        cc_cur.error = error

# dvv = [x.dvv for x in ccfs]

################################################################################
stas = []
for ck in ckeys:
    c1, c2 = ck.split('_')
    stas.extend([c1[:-2], c2[:-2]])

stas = np.unique(stas)

res = session.query(XCorr.time).distinct().all()
times = np.sort([x[0] for x in res])[-nrecent:]

for dtime in times:
    for sta in stas:
        out = session.query(XCorr.dvv).filter(XCorr.time == dtime).filter(XCorr.ckey.like(f"%{sta}%")).filter(XCorr.dvv.isnot(None)).all()
        if len(out) == 0:
            continue
        print(sta, len(out))
        session.add(VelChange(time=dtime, sta=sta, dvv=np.median(out), error=np.std(out)))
    session.commit()


sta = stas[0]
out = session.query(VelChange).filter(VelChange.sta == sta).order_by(VelChange.time.asc()).all()

vals = []
for sta in stas:
    out = session.query(VelChange.dvv).filter(VelChange.sta == sta).order_by(VelChange.time.asc()).all()
    vals.append([x[0] for x in out])

[plt.plot(x) for x in vals]

# out = session.query(XCorr).filter(XCorr.ckey.like(f"%{sta}%")).filter(XCorr.dvv.isnot(None)).all()


#########################################################################################

# out = session.query(XCorr.time).all()
# out = session.query(XCorr.time).group_concat(XCorr.time).all()
# out = session.query(XCorr.time).group_by(XCorr.time).all()
res = session.query(XCorr.time).distinct().all()
times = np.sort([x[0] for x in res])



# out = session.query(XCorr.dvv).filter(XCorr.time == times[1]).filter(XCorr.ckey.like(f"%{sta}%")).filter(XCorr.dvv.isnot(None)).all()
out = session.query(XCorr).filter(XCorr.time == times[1]).filter(XCorr.ckey.like(f"%{sta}%")).filter(XCorr.dvv.isnot(None)).all()


out = session.query(XCorr.dvv).filter(XCorr.ckey.like(f"%{sta}%")).filter(XCorr.dvv.isnot(None)).all()
x = VelChange(sta=sta, dvv=np.median(out), error=np.std(out))

print(np.median(out), np.std(out))
plt.plot(np.array(out))


for sta in stas:
    out = session.query(XCorr.dvv).filter(XCorr.ckey.like(f"%{sta}%")).filter(XCorr.dvv.isnot(None)).all()
    session.add(VelChange(time=time, sta=sta, dvv=np.median(out), error=np.std(out)))
session.commit()


###############################################################################################
##########################################


ckeys = np.unique(session.query(XCorr.ckey).all())

ckeys = np.unique(session.query(XCorr.ckey).all())
# dists = xchange.ckey_dists(ckeys)
ckey = ckeys[0]
dist = session.query(ChanPair.dist).filter(ChanPair.ckey == ck).first()[0]
# dist = dists[ckey]
coda_start_sec = dist / coda_start_vel

print(f"{ckey}: {dist:.2f}m")

# for i, ckey in enumerate(ckeys):

ccfs = session.query(XCorr).filter_by(ckey=ckey).order_by(XCorr.time.desc()).limit(nrecent).all()[::-1]

for icc in range(1, len(ccfs)):
    print(i)
    cc_ref = ccfs[icc - 1]
    cc_cur = ccfs[icc]
    sig_ref = xio.bytes_to_array(rhandle.get(cc_ref.data))
    sig_cur = xio.bytes_to_array(rhandle.get(cc_cur.data))
    dvv, error = xchange.dvv(sig_ref, sig_cur, sr, wlen_sec, whiten_freqs, coda_start_sec, coda_end_sec)
    cc_cur.dvv = dvv
    cc_cur.error = error

dvv = [x.dvv for x in ccfs]


# dc = np.array([xio.bytes_to_array(rhandle.get(x.data)) for x in rows])
# xplot.im(dc)










##########################################################

reload(xchange)
from xseis2.xchange import windowed_fft, measure_shift_fwins_cc, linear_regression_zforce

# outlier_clip = 0.02
# dvv_outlier_clip = 10.0
dvv_outlier_clip = None
interp_factor = 100
cc1 = sig_ref
cc2 = sig_cur
cfreqs = whiten_freqs

assert(len(cc1) == len(cc2))

wlen = int(wlen_sec * sr)
coda_start = int(coda_start_sec * sr)
coda_end = int(coda_end_sec * sr)

hl = len(cc1) // 2
iwin = [hl - coda_end, hl + coda_end]

stepsize = wlen // 4
slices = xutil.build_slice_inds(iwin[0], iwin[1], wlen, stepsize=stepsize)
print("num slices", len(slices))

# %timeit fwins1, filt = xchange.windowed_fft(sig1, slices, sr, cfreqs)
fwins1, filt = windowed_fft(cc1, slices, sr, cfreqs)
fwins2, filt = windowed_fft(cc2, slices, sr, cfreqs)
imax, coh = measure_shift_fwins_cc(fwins1, fwins2, interp_factor=interp_factor).T

print("mean coh %.3f" % np.mean(coh))

xv = np.mean(slices, axis=1) - hl

ik = np.arange(len(xv))

is_coda = np.abs(xv) > coda_start

if dvv_outlier_clip is not None:
    is_outlier = np.abs(imax / xv) < (dvv_outlier_clip / 100)
    ik = np.where((is_coda) & (is_outlier))[0]
    print(f"non-outlier: {np.sum(is_outlier) / len(is_outlier) * 100:.2f}%")
else:
    ik = np.where((is_coda))[0]

# nkeep = (len(ik) / len(xv)) * 100

yint, slope, res = linear_regression_zforce(xv[ik], imax[ik], coh[ik] ** 2)

# print("tt_change: %.5f%% ss_res: %.4e " % (slope * 100, res))

tfit = yint + slope * xv
plt.scatter(xv[ik], imax[ik], c=coh[ik])
plt.colorbar()
mask = np.ones_like(xv, bool)
mask[ik] = False
plt.scatter(xv[mask], imax[mask], c='red', alpha=0.2)
# plt.scatter(xv[ik], imax[ik], c='red', alpha=0.5)
plt.plot(xv, tfit)
# plt.plot(xv, tfit)
plt.title("tt_change: %.3f%% ss_res: %.3f " % (slope * 100, res))
plt.axvline(0, linestyle='--')
alpha = 0.5
# vel = 3200.
# direct = dist / vel * sr
plt.axvline(coda_start, linestyle='--', color='green', alpha=alpha)
plt.axvline(-coda_start, linestyle='--', color='green', alpha=alpha)