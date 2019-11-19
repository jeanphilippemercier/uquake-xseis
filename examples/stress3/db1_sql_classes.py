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

# from microquake.io.h5stream import H5Stream

from sqlalchemy import Table, Column, Integer, String, MetaData, ForeignKey, DateTime, Float, Sequence, ARRAY, LargeBinary
from sqlalchemy.sql import select
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import redis

Base = declarative_base()

plt.ion()

rh = redis.Redis(host='localhost', port=6379, db=0)
rh.flushall()

################################
db_string = "postgres://postgres:postgres@localhost"
db = create_engine(db_string)
print(db.table_names())
Session = sessionmaker(bind=db)
session = Session()
metadata = MetaData(db)
metadata.reflect()
conn = db.connect()

# table = metadata.tables['xcorrs']
# table.drop()
# session.commit()
# conn.execute(table.delete())
# conn.execute(table.drop())


class VelChange(Base):
    __tablename__ = 'velchanges'

    id = Column(Integer, primary_key=True)
    time = Column(DateTime)
    key = Column(String(10))
    # hours = Column(String(10))
    dvv = Column(Float)
    error = Column(Float)

    def __repr__(self):
        return f"<VelChange({self.time}, {self.cey}, {self.dvv}, {self.error}"


class Xcorr(Base):
    __tablename__ = 'xcorrs'

    id = Column(Integer, primary_key=True)
    time = Column(DateTime)
    ckey = Column(String(10))
    data = Column(String(30))
    dvv = Column(Float)
    error = Column(Float)

    def __repr__(self):
        return f"<Xcorr({self.time}, {self.ckey}, {self.data}, {self.dvv}, {self.error}"


Base.metadata.create_all(db)
session.commit()
# table = Base.metadata.tables['xcorrs']

# metadata.create_all(db)


out = Xcorr(time=datetime.now(), ckey='xx')
session.add(out)
out = session.query(Xcorr).filter_by(ckey='xx').first()


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
    conn.execute(table.insert(), dict(time=t0, ckey=ckey, data=dkey))
##################################################


# get ckeys and compute interpair dists
sel = select([table.c.ckey])
out = conn.execute(sel).fetchall()
ckeys = np.unique([v[0] for v in out])
# dists = xchange.ckey_dists(ckeys)

coda_vel = 3200.
n_most_recent = 30

# s = select([table.c.time]).where(table.c.time.between(t0, t1))
sel = select([table.c.data]).where(table.c.ckey == ckey).order_by(table.c.time.desc()).limit(n_most_recent)
out = conn.execute(sel).fetchall()[::-1]

keys = [v[0] for v in out]
dc = np.array([xio.bytes_to_array(rh.get(k)) for k in keys])
xplot.im(dc)



# table = Table('xcorrs', metadata,
# Column('id', Integer, Sequence('id_seq'), primary_key=True),
# Column('time', DateTime),
# Column('ckey', String(10)),
# Column('data', String(30)),
# Column('dvv', Float),
# # Column('data', ARRAY(Float)),
# )


# class User(Base):
#     __tablename__ = 'users'
#     id = Column(sqlalchemy.Integer, primary_key=True)
#     name = Column(sqlalchemy.String)

#     def __init__(self, code=None, *args, **kwargs):
#         self.name = name

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


session.query().filter(table.c.ckey).\
       update({"no_of_logins": (User.no_of_logins +1)})
   session.commit()



# start = time.time()

# for ck in ckeys:
#     print(ck)
#     sel = select([table.c.data]).where(table.c.ckey == ckey).order_by(table.c.time.desc()).limit(5)
#     out = conn.execute(sel).fetchall()
#     keys = [v[0] for v in out]
#     dc = np.array([xio.bytes_to_array(rh.get(k)) for k in keys])

# print(time.time() - start)