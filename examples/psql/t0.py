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
# import time

# # from obspy import read
# # from obspy.io.rg16.core import _read_rg16
# from microquake.core import read
# from spp.core.settings import settings
# from glob import glob
# from importlib import reload

import psycopg2

plt.ion()

conn = psycopg2.connect("dbname=tutorial user=postgres password=postgres host=localhost")
cur = conn.cursor()

cur.execute("INSERT INTO conditions(time, location, temperature, humidity)\
  VALUES (NOW(), 'desk', 20.0, 30.0);")

cur.execute("INSERT INTO conditions(time, location, humidity)\
  VALUES (NOW(), 'roof', 9.0);")

cur.execute("SELECT * FROM conditions ORDER BY time DESC LIMIT 100;")
cur.execute("SELECT humidity FROM conditions ORDER BY time DESC LIMIT 100;")
cur.execute("SELECT temperature FROM conditions ORDER BY time DESC LIMIT 100;")
# cur.fetchone()
cur.fetchall()

# cur.execute("CREATE TABLE sgrams (id serial PRIMARY KEY, num integer, data varchar);")
# cur.execute("CREATE TABLE sgrams (time TIMESTAMPTZ NOT NULL, num integer, data varchar);")

cur.execute("CREATE TABLE sgrams ( \
  time        TIMESTAMPTZ       NOT NULL,\
  chan1       REAL  NULL,\
  chan2       REAL  NULL\
);")

cur.execute("SELECT create_hypertable('sgrams', 'time');")

dat = np.arange(1000, dtype=np.float32)
t0 = datetime.now()
sr = 6000.
dt = 1. / sr
times = [t0 + timedelta(seconds=i * dt) for i in range(len(dat))]

for i in range(len(dat)):
    print(i)
    cur.execute("INSERT INTO sgrams(time, chan1, chan2) VALUES (%s, %s, %s);", (times[i], float(dat[i]), float(dat[i] + 0.5)))
    # cur.execute("INSERT INTO sgrams(time, chan1, chan2) VALUES (NOW(), %s, %s);", (float(dat[i]), float(dat[i] + 0.5)))


cur.execute("SELECT chan1 FROM sgrams ORDER BY time ASC;")
cur.execute("SELECT chan1, chan2 FROM sgrams ORDER BY time ASC;")
# cur.execute("SELECT * FROM sgrams;")
# cur.fetchone()
out = cur.fetchall()
