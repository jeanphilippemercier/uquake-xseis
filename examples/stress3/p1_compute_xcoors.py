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
from spp.pipeline.compute_xcorrs import Processor
from glob import glob

from xseis2.h5stream import H5Stream

from sqlalchemy import Table, Column, Integer, String, MetaData, ForeignKey, DateTime, Float, Sequence, ARRAY, LargeBinary
from sqlalchemy.sql import select
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import redis

from xseis2.xsql import VelChange, XCorr, Base
plt.ion()


# t0 = hs.starttime
starttime = datetime(2019, 7, 10, 16, 0, 0, 13)

stacklen = 2000
# Params to function
times = [starttime + timedelta(seconds=i * stacklen) for i in range(5)]

t0 = times[0]
t1 = t0 + timedelta(seconds=stacklen)

proc = Processor()

for t0 in times:
	print(t0)
	t1 = t0 + timedelta(seconds=stacklen)
	proc.process(t0=t0, t1=t1)

##############################



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
Base.metadata.create_all(db)
# conn = db.connect()

##################################
rh = redis.Redis(host='localhost', port=6379, db=0)
# rh.flushall()
###################################

# inject synthetic xcors
reload(xchange)

# ckey = 'xx'
# wlen_sec = 2.0
# sr = 1000.0
# cfreqs = np.array([50, 60, 400, 450])
# tt_changes = np.linspace(0.0, 0.05, 20)
# ccs = xchange.stretch_sim_ccs(wlen_sec, sr, tt_changes, cfreqs)

# # plt.plot(ccs[0])
# # plt.plot(ccs[-1])

# stacklen = 3600
# times = [datetime.now() + timedelta(seconds=i * stacklen) for i in range(ccs.shape[0])]

# for i, t0 in enumerate(times):
#     # write to postgres and redis
#     dkey = f"{str(t0)} {ckey}"
#     rh.set(dkey, xio.array_to_bytes(ccs[i]))
#     session.add(XCorr(time=t0, ckey=ckey, data=dkey))
# ##################################################

out = session.query(XCorr.ckey).filter_by(ckey='xx').all()
out = session.query(XCorr.ckey).all()


nfetch = 30
# out = session.query(XCorr).filter_by(ckey='xx').all()
rows = session.query(XCorr).filter_by(ckey='xx').order_by(XCorr.time.desc()).limit(20).all()[::-1]

dc = np.array([xio.bytes_to_array(rh.get(x.data)) for x in rows])

xplot.im(dc)

