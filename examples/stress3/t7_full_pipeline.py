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
from xseis2.xsql import VelChange, XCorr, Base

import redis
plt.ion()

##################################
ddir = os.path.join(os.environ['SPP_COMMON'], 'ot_cont')
# hfname = os.path.join(ddir, '10hr_1000hz.h5')
hfname = os.path.join(ddir, '10hr_sim.h5')
hstream = H5Stream(hfname)


################################
db_string = "postgres://postgres:postgres@localhost"
db = create_engine(db_string)
print(db.table_names())
# Session = sessionmaker(bind=db)
# session = Session()
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
# stacklen = 3600.0
stacklen = 1000

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

ckeys = np.unique(session.query(XCorr.ckey).all())
dists = xchange.ckey_dists(ckeys)
ckey = ckeys[0]

nfetch = 30
# out = session.query(XCorr).filter_by(ckey='xx').all()
rows = session.query(XCorr).filter_by(ckey=ckey).order_by(XCorr.time.desc()).limit(20).all()[::-1]

dc = np.array([xio.bytes_to_array(rhandle.get(x.data)) for x in rows])

xplot.im(dc)


coda_vel = 3200.
n_most_recent = 30

########################################

sr = 1000.0
vel = 3200.
# dist = dists[ckey]
dist = 300
coda_start = int(dist / vel * sr + 0.5)
coda_end = int(0.8 * sr)
wlen = 50
cfreqs = np.array([80, 100, 250, 300])
print(f"{ckey}: {dist:.2f}m")

#################################
ix = 5
cc1 = dc[0]
cc2 = dc[ix]
# plt.plot(sig1)
# plt.plot(sig2)
print(tt_changes[ix])

out = []

for i, cc2 in enumerate(dc):
    meas = xchange.dvv(cc1, cc2, sr, wlen, cfreqs, coda_start, coda_end)
    out.append(meas)
out = np.array(out)

plt.plot(out[:, 0])
plt.plot(tt_changes)

for i, xc in enumerate(rows):
    dvv, err = out[i]
    xc.dvv = dvv
    xc.error = err


rows = session.query(XCorr.dvv).filter_by(ckey='xx').order_by(XCorr.time.desc()).all()




