import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
# import os

import time
from xseis2 import xio

# import psycopg2
# from psycopg2.extras import execute_values
# from psycopg2.sql import SQL, Identifier
from sqlalchemy import Table, Column, Integer, String, MetaData, ForeignKey, DateTime, Float, Sequence, ARRAY, LargeBinary
from sqlalchemy.sql import select
from sqlalchemy import create_engine
import redis

from sqlalchemy.orm import sessionmaker

plt.ion()


# db_string = "postgres://admin:donotusethispassword@aws-us-east-1-portal.19.dblayer.com:15813/compose"
db_string = "postgres://postgres:postgres@localhost"
db = create_engine(db_string)
print(db.table_names())
conn = db.connect()

Session = sessionmaker(bind=db)
session = Session()

metadata = MetaData(db)
metadata.reflect()

# metadata.bind = db
# conn.execute("DROP TABLE IF EXISTS xcorrs;")

# for table in reversed(meta.Base.metadata.sorted_tables): meta.Session.execute(table.delete()); meta.Session.commit()

for table in reversed(metadata.sorted_tables):
    # db.execute(tbl.delete())
    table.drop(db)
    # db.execute(tbl.drop())
    # session.execute(tbl.drop())
    session.commit()

# metadata.bind = db
# metadata.create_all(db)

# metadata.drop_all(db)
# table = metadata.tables['users'].drop()

table = Table('xcorrs', metadata,
# Column('id', Integer, Sequence('id_seq'), primary_key=True),
Column('id', Integer, primary_key=True),
Column('time', DateTime),
Column('ckey', String(20)),
Column('data', String(50)),
Column('dvv', Float),
# Column('data', ARRAY(Float)),
)
# table.drop(db)
metadata.create_all(db)
session.commit()

#########################################
r = redis.Redis(host='localhost', port=6379, db=0)
r.flushall()
# nchan = 100
# nsamp = 6000 * 10 * 3
nchan = 10
nsamp = 3000 * 1
dat = np.zeros((nchan, nsamp), dtype=float)
for i in range(dat.shape[0]):
    dat[i] = np.arange(dat.shape[1]) + i

# chans = ["chan%d" % i for i in range(nchan)]
chans = np.arange(nchan).astype(str)
tnow = datetime.now()
# sr = 6000.
dt = 3600.
times = [tnow + timedelta(seconds=i * dt) for i in range(12)]

############################
start = time.time()

for tkey in times:
    print(tkey)
    pipe = r.pipeline()

    vals = []
    for i, sig in enumerate(dat):
        print(i)
        # key = f"{str(t0)}_{ck[0]}_{ck[1]}"
        # dkey = f"{str(t0)}_{chans[i]}"
        dkey = f"{int(tkey.timestamp() * 1e6)}_{chans[i]}"
        pipe.set(dkey, xio.array_to_bytes(sig))
        # d = dict(time=tstamp, ckey=chans[i], data=sig.tobytes())
        d = dict(time=tkey, ckey=chans[i], data=dkey, dvv=i)
        vals.append(d)
    pipe.execute()
    conn.execute(table.insert(), vals)

print(time.time() - start)


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



# thirty_days_ago = datetime.today() - timedelta(days=30)
# fifteen_days_ago = datetime.today() - timedelta(days=15)

# Using and_ IMPLICITLY:
s = session.query(table).filter(table.c.time >= t0, table.c.time <= t1).all()

qry = session.query(table).filter(table.time.between(t0, t1))

s = select([table.c.time])
s = select([table.c.time]).where(table.c.time.between(t0, t1))
result = conn.execute(s)
out = result.fetchall()
out
s = select([table.c.time]).where(table.c.ckey=='993')



