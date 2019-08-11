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
r = redis.Redis(host='localhost', port=6379, db=0)
r.flushall()


####################################################

# hs.get_row_indexes(chans[1:10])

# PARAMS - config
whiten_freqs = np.array([60, 80, 320, 350])
cclen = 20.0
keeplag = 1.0
stepsize = cclen
onebit = True
# stacklen = 3600.0
stacklen = 1000

# Params to function
chans = hs.channels[:]
times = [hs.starttime + timedelta(seconds=i * stacklen) for i in range(5)]

# t0 = hs.starttime

for t0 in times:
    print(t0)
    t1 = t0 + timedelta(seconds=stacklen)
    reload(xchange)
    dc, ckeys_ix = xchange.xcorr_stack_slices(hs, t0, t1, chans, cclen, keeplag, whiten_freqs, onebit=onebit)
    ckeys = [f"{ck[0]}_{ck[1]}" for ck in chans[ckeys_ix]]

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

