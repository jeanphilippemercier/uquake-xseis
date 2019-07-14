import matplotlib.pyplot as plt

import numpy as np
# import datetime
# import struct
from datetime import datetime, timedelta
# from obspy.core import UTCDateTime
# import os
# import glob
# # import pickle
# from xseis2 import xutil
# # from xseis import xio
# import h5py
import time

# # from obspy import read
# # from obspy.io.rg16.core import _read_rg16
# from microquake.core import read
# from spp.core.settings import settings
# from glob import glob
# from importlib import reload

import psycopg2
from psycopg2.extras import execute_values
# from psycopg2.extras import SQL
from psycopg2.sql import SQL, Identifier


plt.ion()

conn = psycopg2.connect("dbname=tutorial user=postgres password=postgres host=localhost")
cur = conn.cursor()

# cur.execute('DROP TABLE "sgrams";')
cur.execute('DROP TABLE IF EXISTS sgrams;')

cur.execute("CREATE TABLE sgrams (time TIMESTAMPTZ NOT NULL);")
cur.execute("SELECT create_hypertable('sgrams', 'time');")

nchan = 300
nsamp = 6000 * 1
dat = np.zeros((nchan, nsamp), dtype=float)
for i in range(dat.shape[0]):
    dat[i] = np.arange(dat.shape[1]) + i


t0 = datetime.now()
sr = 6000.
dt = 1. / sr
times = [t0 + timedelta(seconds=i * dt) for i in range(nsamp)]

chans = ["chan%d" % i for i in range(nchan)]

for name in chans:
    cur.execute("ALTER TABLE sgrams ADD COLUMN %s REAL[];" % (name))


cur.execute(SQL("INSERT INTO sgrams(time, {}) VALUES (%s, %s);").format(Identifier(chans[0])), (times[0], list(dat[0])))


start = time.time()

for i, sig in enumerate(dat):
    # cur.execute("INSERT INTO sgrams(time, chan1) VALUES (%s, %s);", (times[0], list(sig)))
    cur.execute(SQL("INSERT INTO sgrams(time, {}) VALUES (%s, %s);").format(Identifier(chans[i])), (times[0], list(sig)))
end = time.time()
print(end - start)


# SQL("INSERT INTO {} VALUES (%s)").format(Identifier('numbers')),
# SQL("INSERT INTO sgrams(time, {}) VALUES (%s, %s);").format(Identifier(chans[0]))


cur.execute("SELECT chan1 FROM sgrams ORDER BY time ASC;")
cur.execute("SELECT chan1, chan2 FROM sgrams ORDER BY time ASC;")
cur.execute("SELECT chan288, chan289 FROM sgrams ORDER BY time ASC;")
# cur.execute("SELECT * FROM sgrams;")
# cur.fetchone()
out = cur.fetchall()
out
sig = out[0][0]

###################
start = time.time()

cur.execute("SELECT * FROM sgrams;")
out = cur.fetchall()

end = time.time()
eps = end - start
print(eps)
