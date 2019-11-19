from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload

from xseis2 import xchange
from xseis2 import xplot
from xseis2 import xio
# from xseis2.xchange import smooth, nextpow2, getCoherence
# from microquake.core import read
# from obspy.core import UTCDateTime

from sqlalchemy import MetaData
from sqlalchemy.sql import select
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import redis
plt.ion()


def ckey_dists(ckeys):

    sites = [station.code for station in settings.inventory.stations()]
    site_locs = [station.loc for station in settings.inventory.stations()]
    ldict = dict(zip(sites, site_locs))

    dists = dict()
    for ck in ckeys:
        c1, c2 = ck.split('_')
        dd = xutil.dist(ldict[c1[:-2]], ldict[c2[:-2]])
        dists[ck] = dd
    return dists


################################
db_string = "postgres://postgres:postgres@localhost"
db = create_engine(db_string)
print(db.table_names())
conn = db.connect()
Session = sessionmaker(bind=db)
session = Session()
metadata = MetaData(db)
metadata.reflect()
table = metadata.tables['xcorrs']
# conn.execute(table.delete())
# session.commit()

##################################
rh = redis.Redis(host='localhost', port=6379, db=0)
# r.flushall()


###################################


# get ckeys and compute interpair dists
sel = select([table.c.ckey])
out = conn.execute(sel).fetchall()
ckeys = np.unique([v[0] for v in out])
dists = ckey_dists(ckeys)

coda_vel = 3200.
n_most_recent = 30


# load data for one pair
ckey = ckeys[30]
ckey = 'test3'
# sel = select([table.c.data]).where(table.c.ckey == ckey)
# sel = select([table.c.data]).where(table.c.ckey == ckey).order_by(table.c.time.desc()).limit(5)
sel = select([table.c.data]).where(table.c.ckey == ckey).order_by(table.c.time.desc()).limit(n_most_recent)
out = conn.execute(sel).fetchall()[::-1]

keys = [v[0] for v in out]
dc = np.array([xio.bytes_to_array(rh.get(k)) for k in keys])
xplot.im(dc)

# s = select([table.c.time]).where(table.c.time.between(t0, t1))

########################################

sr = 1000.0
vel = 3200.
# dist = dists[ckey]
dist = 300
coda_start = int(dist / vel * sr + 0.5)
coda_end = int(0.8 * sr)
print(f"{ckey}: {dist:.2f}m")

#################################
ix = 5
sig1 = dc[0]
# sig2 = dc[-1]
sig2 = dc[ix]
# plt.plot(sig1)
# plt.plot(sig2)
print(tt_changes[ix])

# print(f"dist {dist:.2f} m")

hl = len(sig1) // 2
cfreqs = np.array([80, 100, 250, 300])
# cfreqs = np.array([80, 100, 300, 350])

iwin = [hl - coda_end, hl + coda_end]
# iwin = [hl - 2000, hl + 2000]

wlen = 50
stepsize = wlen // 4
slices = xutil.build_slice_inds(iwin[0], iwin[1], wlen, stepsize=stepsize)
nslice = len(slices)
print("num slices", len(slices))
reload(xchange)

# %timeit fwins1, filt = xchange.windowed_fft(sig1, slices, sr, cfreqs)
fwins1, filt = xchange.windowed_fft(sig1, slices, sr, cfreqs)
fwins2, filt = xchange.windowed_fft(sig2, slices, sr, cfreqs)
imax, coh = xchange.measure_shift_fwins_cc(fwins1, fwins2, interp_factor=100).T

print("mean coh %.3f" % np.mean(coh))

xv = np.mean(slices, axis=1) - hl

outlier_val = 0.002
is_out = np.abs(imax / xv) < outlier_val
ik = np.where((np.abs(xv) > coda_start) & (np.abs(imax / xv) < outlier_val))[0]
# nkeep = (len(ik) / len(xv)) * 100
print(f"non-outlier: {np.sum(is_out) / len(is_out) * 100:.2f}%")

yint, slope, res = xchange.linear_regression4(xv[ik], imax[ik], coh[ik] ** 2)

print("tt_change: %.5f%% ss_res: %.4e " % (slope * 100, res))
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


########################################


import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from importlib import reload
from scipy import fftpack
from numpy.fft import fft, ifft, rfft, fftfreq

from scipy import interpolate
import scipy.signal

from xseis2 import xutil
from xseis2 import xchange
from xseis2 import xplot
# from xseis2.xchange import smooth, nextpow2, getCoherence
from numpy.polynomial.polynomial import polyfit


wlen_sec = 1.0
sr = 1000.0

ncc = 20
reload(xchange)
nsamp = int(wlen_sec * sr)
nsamp_cc = int(wlen_sec * sr * 2)

# tt_change_percent = 0.8
# tt_change_percent = 0.05
tt_changes = np.linspace(0.0, 0.05, ncc)

# time = np.arange(nsamp) / sr

cfreqs = [70, 100, 350, 400]
ref_half = xutil.noise1d(nsamp, cfreqs, sr, scale=1, taplen=0.0)

ccs = np.zeros((ncc, nsamp_cc), dtype=np.float32)

for i, tt_change in enumerate(tt_changes):
    tmp = xchange.stretch(ref_half, sr, tt_change)
    cc = np.concatenate((tmp[::-1], tmp))
    ccs[i] = cc

# xplot.im(ccs)

plt.plot(ccs[0])
plt.plot(ccs[-1])

stacklen = 3600
times = [datetime.now() + timedelta(seconds=i * stacklen) for i in range(ccs.shape[0])]

for i, t0 in enumerate(times):
    # write to postgres and redis
    ckey = 'test3'
    dkey = f"{str(t0)} {ckey}"
    rh.set(dkey, xio.array_to_bytes(ccs[i]))
    conn.execute(table.insert(), dict(time=t0, ckey=ckey, data=dkey))


# start = time.time()

# for ck in ckeys:
#     print(ck)
#     sel = select([table.c.data]).where(table.c.ckey == ckey).order_by(table.c.time.desc()).limit(5)
#     out = conn.execute(sel).fetchall()
#     keys = [v[0] for v in out]
#     dc = np.array([xio.bytes_to_array(rh.get(k)) for k in keys])

# print(time.time() - start)
