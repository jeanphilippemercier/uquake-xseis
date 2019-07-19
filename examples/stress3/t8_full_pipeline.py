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
from xseis2 import xchange_workflow as flow

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
flow.sql_drop_tables(db)
session.commit()

Base.metadata.create_all(db)
session.commit()


##################################
rhandle = redis.Redis(host='localhost', port=6379, db=0)
rhandle.flushall()
###################################

flow.fill_table_chanpairs(session)
flow.fill_table_xcorrs(hstream, session, rhandle)
flow.measure_dvv_xcorrs(session, rhandle)
flow.measure_dvv_stations(session)


stas = np.unique(session.query(VelChange.sta).all())


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
