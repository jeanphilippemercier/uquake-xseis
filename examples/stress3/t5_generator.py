from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
import os
from importlib import reload

from xseis2 import xchange
from xseis2 import xplot
from xseis2 import xio
# from xseis2.xchange import smooth, nextpow2, getCoherence
# from microquake.core import read
# from obspy.core import UTCDateTime

from microquake.io.h5stream import H5Stream

from sqlalchemy.sql import select

plt.ion()


########################################

ddir = os.path.join(os.environ['SPP_COMMON'], 'ot_cont')
# hfname = os.path.join(ddir, '10hr_1000hz.h5')
hfname = os.path.join(ddir, '10hr_sim.h5')
hs = H5Stream(hfname)
locs = hs.hf['locs'][:]
chans = hs.channels
locs = locs[hs.get_row_indexes(chans)]
sr = hs.samplerate
####################################################

# hs.get_row_indexes(chans[1:10])
# PARAMS - config
whiten_freqs = np.array([60, 80, 320, 350])
cclen = 10.0
keeplag = 1.0
stepsize = cclen
onebit = True
stacklen = 3600.0
# stacklen = 1000

# Params to function
chans = hs.channels[:]
times = [hs.starttime + timedelta(seconds=i * stacklen) for i in range(5)]

# t0 = hs.starttime

t0 = times[0]
t1 = times[1]
print(t0)
reload(xchange)

datgen = hs.slice_gen(t0, t1, chans, cclen)

reload(xchange)
dc, ckeys_ix = xchange.xcorr_stack_slices(datgen, chans, cclen, sr, keeplag, whiten_freqs, onebit=onebit)
# dc, ckeys_ix = xchange.xcorr_stack_slices2(hs, t0, t1, chans, cclen, keeplag, whiten_freqs, onebit=onebit)
ckeys = [f"{ck[0]}_{ck[1]}" for ck in chans[ckeys_ix]]

xplot.im(dc)

# np.allclose(a, dc)

plt.plot(a[0])
plt.plot(dc[0])

pipe = r.pipeline()
vals = []
for i, sig in enumerate(dc):
    # print(i)
    ckey = ckeys[i]
    dkey = f"{str(t0)} {ckey}"
    pipe.set(dkey, xio.array_to_bytes(sig))
    # d = dict(time=tstamp, ckey=chans[i], data=sig.tobytes())
    d = dict(time=t0, ckey=ckey, data=dkey, dvv=i)
    vals.append(d)
conn.execute(table.insert(), vals)
pipe.execute()


###################################
# t0 = times[2]
# t1 = times[5]
sel = select([table.c.data]).where(table.c.ckey == ckey)
# s = select([table.c.time]).where(table.c.time.between(t0, t1))
result = conn.execute(sel)
out = result.fetchall()
out
########################################

keys = [v[0] for v in out]
dc = np.array([xio.bytes_to_array(r.get(k)) for k in keys])
xplot.im(dc)

