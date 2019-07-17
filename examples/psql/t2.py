import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
# import os

import time

import psycopg2
from psycopg2.extras import execute_values
from psycopg2.sql import SQL, Identifier


plt.ion()

conn = psycopg2.connect("dbname=tutorial user=postgres password=postgres host=localhost")
cur = conn.cursor()

# cur.execute('DROP TABLE "sgrams";')
cur.execute('DROP TABLE IF EXISTS sgrams;')

cur.execute("CREATE TABLE sgrams (time TIMESTAMPTZ PRIMARY KEY NOT NULL);")
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


cur.execute("select * FROM sgrams")
colnames = [desc[0] for desc in cur.description]

start = time.time()
i = 2
cur.execute(SQL("INSERT INTO sgrams(time, {}) VALUES (%s, %s);").format(Identifier(chans[i])), (times[0], list(dat[i])))
end = time.time()
print(end - start)

start = time.time()

for i, sig in enumerate(dat):
    print(i)
    cur.execute(SQL("INSERT INTO sgrams(time, {}) VALUES (%s, %s);").format(Identifier(chans[i])), (times[0], list(dat[i])))
    # cur.execute("INSERT INTO sgrams(time, chan1) VALUES (%s, %s);", (times[0], list(sig)))
    # cur.execute(SQL("INSERT INTO sgrams(time, {}) VALUES (%s, %s);").format(Identifier(chans[i])), (times[0], list(sig)))
end = time.time()
print(end - start)


# SQL("INSERT INTO {} VALUES (%s)").format(Identifier('numbers')),
# SQL("INSERT INTO sgrams(time, {}) VALUES (%s, %s);").format(Identifier(chans[0]))


cur.execute("SELECT chan0 FROM sgrams ORDER BY time ASC;")
cur.execute("SELECT chan298 FROM sgrams ORDER BY time ASC;")
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
