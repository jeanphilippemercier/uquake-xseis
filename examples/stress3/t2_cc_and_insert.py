import os
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from importlib import reload

from xseis2 import xutil
from xseis2 import xchange
from xseis2 import xplot
from xseis2 import xio
# from xseis2.xchange import smooth, nextpow2, getCoherence
# from microquake.core import read
# from microquake.core import UTCDateTime
from spp.core.settings import settings
from glob import glob

from microquake.io.h5stream import H5Stream

from sqlalchemy import Table, Column, Integer, String, MetaData, ForeignKey, DateTime, Float, Sequence, ARRAY, LargeBinary
from sqlalchemy.sql import select
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import redis


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
dc, ckeys_ix = xchange.xcorr_stack_slices(hs, t0, t1, chans, cclen, keeplag, whiten_freqs, onebit=onebit)


ckeys = chans[ckeys_ix]
tkey = t0.datetime
# tkey_epoch = int(t0.datetime.timestamp() * 1e6)


locs = hs.hf['locs'][:]
locs = locs[hs.get_row_indexes(chans)]
dd = xutil.dist_diff_ckeys(ckeys_ix, locs)
isort = np.argsort(dd)
# xplot.im(dc[isort])
xplot.im(dc[isort], norm=False)

xplot.im(dc)

plt.plot(dc[0])

###################################

import time
db_string = "postgres://postgres:postgres@localhost"
db = create_engine(db_string)
print(db.table_names())
conn = db.connect()

Session = sessionmaker(bind=db)
session = Session()
metadata = MetaData(db)
metadata.reflect()

table = metadata.tables['xcorrs']

r = redis.Redis(host='localhost', port=6379, db=0)
r.flushall()

start = time.time()

print(tkey)
pipe = r.pipeline()

vals = []
for i, sig in enumerate(dc):
    print(i)
    ckey = f"{ckeys[i][0]}_{ckeys[i][1]}"
    dkey = f"{str(tkey)} {ckey}"
    pipe.set(dkey, xio.array_to_bytes(sig))
    # d = dict(time=tstamp, ckey=chans[i], data=sig.tobytes())
    d = dict(time=tkey, ckey=ckey, data=dkey, dvv=i)
    vals.append(d)
conn.execute(table.insert(), vals)
pipe.execute()

print(time.time() - start)
#############################################



t0 = times[2]
t1 = times[5]

s = select([table.c.time]).where(table.c.time.between(t0, t1))
s = select([table.c.data]).where(table.c.ckey == chans[2])
result = conn.execute(s)
out = result.fetchall()
out
########################################

keys = [x[0] for x in out]
out = []
start = time.time()

for i, key in enumerate(keys):
    out.append(xio.bytes_to_array(r.get(key)))
print(time.time() - start)
