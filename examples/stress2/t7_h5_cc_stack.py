from datetime import timedelta

import numpy as np

import matplotlib.pyplot as plt
import os
from importlib import reload

from xseis2 import xutil
from xseis2 import xchange
from xseis2 import xplot
from xseis2 import xio
# from xseis2.xchange import smooth, nextpow2, getCoherence
# from microquake.core import read

from microquake.io.h5stream import H5Stream

# from xseis2.xutil import bandpass, freq_window, phase

plt.ion()

ddir = os.path.join(os.environ['SPP_COMMON'], 'ot_cont')
# hfname = os.path.join(ddir, '10hr_1000hz.h5')
hfname = os.path.join(ddir, '10hr_sim.h5')

hs = H5Stream(hfname)
hs.starttime
hs.endtime
chans = hs.channels
# hs.get_row_indexes(chans[1:10])

# PARAMS - config
whiten_freqs = np.array([60, 80, 320, 350])
cclen = 10.0
keeplag = 1.0
stepsize = cclen
onebit = True

# Params to function

t0 = hs.starttime
# t1 = t0 + timedelta(hours=1)
t1 = t0 + timedelta(hours=0.2)
# dat = hs.query(chans, t0, t1)
# def compute_cc_stack(t0, t1, channels=None):

reload(xutil)

# chans = hs.channels[0:10]
chans = hs.channels[:]
# ckeys = xutil.unique_pairs(np.arange(len(chans)))
# ckeys = xutil.ckeys_remove_intersta(ckeys, chans)
reload(xchange)
dc, ckeys = xchange.xcorr_stack_slices(hs, t0, t1, chans, cclen, keeplag, whiten_freqs, onebit=onebit)

locs = hs.hf['locs'][:]
locs = locs[hs.get_row_indexes(chans)]
dd = xutil.dist_diff_ckeys(ckeys, locs)
isort = np.argsort(dd)
# xplot.im(dc[isort])
xplot.im(dc[isort], norm=False)

xplot.im(dc)

plt.plot(dc[0])

###################################

import time
import redis
# Redis connection
r = redis.Redis(host='localhost', port=6379, db=0)
r.flushdb()

keys = chans[ckeys]
ck = keys[0]
key = f"{str(t0)}_{ck[0]}_{ck[1]}"

start = time.time()

for i, ck in enumerate(keys):
    key = f"{str(t0)}_{ck[0]}_{ck[1]}"

    r.set(key, xio.array_to_bytes(dc[i]))

print(time.time() - start)


keys = np.array(r.keys("*_112.X*"), dtype=str)
times = [x.split('_')[0] for x in keys]
# %timeit keys = r.keys("*_298")
# %timeit keys = r.keys("0_*")

out = []
start = time.time()

for i, key in enumerate(keys):
    out.append(xio.bytes_to_array(r.get(key)))
print(time.time() - start)

out = np.array(out)

xplot.im(out)

# dat = np.arange(60000, dtype=np.float32)
# buf = array_to_bytes(dat)
# buf = dat.tobytes()
# r.set(key, buf)
