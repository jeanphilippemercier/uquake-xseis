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
plt.ion()


########################################

ddir = os.path.join(os.environ['SPP_COMMON'], 'ot_cont')
# hfname = os.path.join(ddir, '10hr_1000hz.h5')
hfname = os.path.join(ddir, '10hr_sim.h5')
hs = H5Stream(hfname)
locs = hs.hf['locs'][:]
chans = hs.channels
sr = hs.samplerate
locs = locs[hs.get_row_indexes(chans)]


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
conn.execute(table.delete())
session.commit()

##################################
rh = redis.Redis(host='localhost', port=6379, db=0)
rh.flushall()


####################################################

# hs.get_row_indexes(chans[1:10])

# PARAMS - config
whiten_freqs = np.array([60, 80, 320, 350])
cclen = 20.0
keeplag = 1.0
stepsize = cclen
onebit = True
# stacklen = 3600.0
stacklen = 2000

# Params to function
chans = hs.channels[:]
times = [hs.starttime + timedelta(seconds=i * stacklen) for i in range(5)]

for t0 in times:
    print(t0)
    t1 = t0 + timedelta(seconds=stacklen)
    # dc, ckeys_ix = xchange.xcorr_stack_slices(hs, t0, t1, chans, cclen, keeplag, whiten_freqs, onebit=onebit)
    datgen = hs.slice_gen(t0, t1, chans, cclen)
    dc, ckeys_ix = xchange.xcorr_stack_slices(datgen, chans, cclen, sr, keeplag, whiten_freqs, onebit=onebit)
    ckeys = [f"{ck[0]}_{ck[1]}" for ck in chans[ckeys_ix]]

    # write to postgres and redis
    pipe = rh.pipeline()
    bulk = []
    for i, sig in enumerate(dc):
        # print(i)
        ckey = ckeys[i]
        dkey = f"{str(t0)} {ckey}"
        pipe.set(dkey, xio.array_to_bytes(sig))
        bulk.append(dict(time=t0, ckey=ckey, data=dkey))
    conn.execute(table.insert(), bulk)
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
dc = np.array([xio.bytes_to_array(rh.get(k)) for k in keys])
xplot.im(dc)