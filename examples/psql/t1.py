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

plt.ion()

conn = psycopg2.connect("dbname=tutorial user=postgres password=postgres host=localhost")
cur = conn.cursor()

# cur.execute('DROP TABLE "sgrams";')
cur.execute('DROP TABLE IF EXISTS sgrams;')

cur.execute("CREATE TABLE sgrams (time TIMESTAMPTZ PRIMARY KEY NOT NULL);")
# cur.execute("SELECT create_hypertable('sgrams', 'time');")

nchan = 300
nsamp = 6000 * 1
dat = np.zeros((nchan, nsamp), dtype=np.float32)
for i in range(dat.shape[0]):
    dat[i] = np.arange(dat.shape[1]) + i


t0 = datetime.now()
sr = 6000.
dt = 1. / sr
times = [t0 + timedelta(seconds=i * dt) for i in range(nsamp)]

chans = np.arange(nchan).astype(str)

for name in chans:
    cur.execute("ALTER TABLE sgrams ADD COLUMN %s REAL;" % ("chan" + name))


start = time.time()

values = []
for i, tdat in enumerate(dat.T):
    values.append(tuple([times[i]] + list(tdat.astype(float))))
end = time.time()
eps = end - start
print(eps)


start = time.time()

execute_values(cur, 'INSERT INTO sgrams VALUES %s', values)

end = time.time()
eps = end - start
print(eps)




cur.execute("CREATE TABLE sgrams (time TIMESTAMPTZ PRIMARY KEY NOT NULL);")
cur.execute("SELECT create_hypertable('sgrams', 'time');")

#....#

for name in column_names:
    cur.execute("ALTER TABLE sgrams ADD COLUMN %s REAL;" % (name))
#....#

# inserts 1 second of data (6000 rows of 300 values each)
execute_values(cur, 'INSERT INTO sgrams VALUES %s', data)



# cur.execute("SELECT create_hypertable('sgrams', 'time');")

# for i in range(len(dat)):
#     print(i)
#     cur.execute("INSERT INTO sgrams(time, chan1, chan2) VALUES (%s, %s, %s);", (times[i], float(dat[i]), float(dat[i] + 0.5)))
#     # cur.execute("INSERT INTO sgrams(time, chan1, chan2) VALUES (NOW(), %s, %s);", (float(dat[i]), float(dat[i] + 0.5)))


cur.execute("SELECT chan1 FROM sgrams ORDER BY time ASC;")
cur.execute("SELECT chan1, chan2 FROM sgrams ORDER BY time ASC;")
# cur.execute("SELECT * FROM sgrams;")
# cur.fetchone()
out = cur.fetchall()
out

###################
start = time.time()

cur.execute("SELECT * FROM sgrams;")
out = cur.fetchall()

end = time.time()
eps = end - start
print(eps)
