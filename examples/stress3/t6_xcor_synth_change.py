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

# from microquake.io.h5stream import H5Stream

from sqlalchemy import MetaData
from sqlalchemy.sql import select
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sqlalchemy import create_engine
from xseis2.xsql import VelChange, XCorr, Base

import redis
plt.ion()

################################
db_string = "postgres://postgres:postgres@localhost"
db = create_engine(db_string)
print(db.table_names())
Session = sessionmaker(bind=db)
session = Session()

metadata = MetaData(db)
metadata.reflect()

for table in reversed(metadata.sorted_tables):
    # db.execute(tbl.delete())
    table.drop(db)
    # session.execute(tbl.drop())
    session.commit()

Base.metadata.create_all(db)
# conn = db.connect()


##################################
rh = redis.Redis(host='localhost', port=6379, db=0)
rh.flushall()
###################################

# inject synthetic xcors
reload(xchange)

ckey = 'xx'
wlen_sec = 2.0
sr = 1000.0
cfreqs = np.array([50, 60, 400, 450])
tt_changes = np.linspace(0.0, 0.05, 20)
ccs = xchange.stretch_sim_ccs(wlen_sec, sr, tt_changes, cfreqs)

# plt.plot(ccs[0])
# plt.plot(ccs[-1])

stacklen = 3600
times = [datetime.now() + timedelta(seconds=i * stacklen) for i in range(ccs.shape[0])]

for i, t0 in enumerate(times):
    # write to postgres and redis
    dkey = f"{str(t0)} {ckey}"
    rh.set(dkey, xio.array_to_bytes(ccs[i]))
    session.add(XCorr(time=t0, ckey=ckey, data=dkey))
##################################################

nfetch = 30
# out = session.query(XCorr).filter_by(ckey='xx').all()
rows = session.query(XCorr).filter_by(ckey='xx').order_by(XCorr.time.desc()).limit(20).all()[::-1]

dc = np.array([xio.bytes_to_array(rh.get(x.data)) for x in rows])

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




# start = time.time()

# for ck in ckeys:
#     print(ck)
#     sel = select([table.c.data]).where(table.c.ckey == ckey).order_by(table.c.time.desc()).limit(5)
#     out = conn.execute(sel).fetchall()
#     keys = [v[0] for v in out]
#     dc = np.array([xio.bytes_to_array(rh.get(k)) for k in keys])

# print(time.time() - start)
