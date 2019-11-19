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
# from xseis2 import xplot
# from spp.core.settings import settings
from glob import glob

from xseis2.h5stream import H5Stream

from sqlalchemy import MetaData
from sqlalchemy.sql import select
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sqlalchemy import create_engine
# from xseis2.xsql import VelChange, XCorr, ChanPair, Base
# from xseis2 import xchange_workflow as flow

from microquake.core.helpers.timescale_db import get_continuous_data
from microquake.core.settings import settings


import redis

# import xseis2
# xseis2.__file__

# sudo wg-quick up philipped
# kubectl exec -it spp-phil bash
# ipython

# plt.ion()
# xio.testf()
##################################
# ddir = os.path.join(os.environ['SPP_COMMON'], 'ot_cont')
# hfname = os.path.join(ddir, '10hr_1000hz.h5')
# hfname = os.path.join(ddir, '10hr_sim.h5')
# hstream = H5Stream(hfname)



# flow.fill_table_chanpairs(session)
# flow.fill_table_xcorrs(hstream, session, rhandle)
# flow.measure_dvv_xcorrs(session, rhandle)
# flow.measure_dvv_stations(session)

from pytz import utc
# starttime

import time
t0 = time.time()


curtime = datetime.utcnow()
curtime.replace(tzinfo=utc)
endtime = curtime + timedelta(minutes=-120)
starttime = endtime - timedelta(minutes=1)
st = get_continuous_data(starttime, endtime)

elapsed = time.time() - t0
print('elapsed: %.2f sec' % elapsed)

[tr.stats.station for tr in st]
[len(tr.data) / 6000 for tr in st]

sensor_id = '1'
st = get_continuous_data(starttime, endtime, sensor_id)


curtime = datetime.utcnow()
curtime.replace(tzinfo=utc)
endtime = curtime + timedelta(minutes=-60)
starttime = endtime - timedelta(minutes=5)
print(starttime, endtime)
st = get_continuous_data(starttime, endtime)


tds = np.arange(-12, 12, 1) * 60

for td in tds:
	endtime = curtime + timedelta(minutes=int(td))
	starttime = endtime - timedelta(minutes=5)
	print(starttime, endtime)
	st = get_continuous_data(starttime, endtime)
	if st is not None:
		break


#########################################################





from microquake.core.stream import Trace, Stream
from microquake.core.settings import settings
import numpy as np
from obspy.core import UTCDateTime
from loguru import logger
from microquake.db.models.alchemy import ContinuousData
from microquake.db.connectors import connect_timescale
from datetime import datetime
from sqlalchemy import desc
from pytz import utc


curtime = datetime.utcnow()
curtime.replace(tzinfo=utc)
endtime = curtime + timedelta(minutes=-120)
starttime = endtime - timedelta(minutes=1)
# st = get_continuous_data(starttime, endtime)


session, engine = connect_timescale()
inventory = settings.inventory

e_time = endtime
s_time = starttime

network_code = inventory.networks[0].code

t = ContinuousData.time
et = ContinuousData.end_time
sid = ContinuousData.sensor_id

import time
start_time = time.time()

cds = session.query(ContinuousData).filter(t <= e_time, et > s_time)

elapsed = time.time() - start_time
print('elapsed: %.2f sec' % elapsed)

traces = []
for cd in cds:
    print(cd)
    x = np.array(cd.x)
    y = np.array(cd.y)
    z = np.array(cd.z)







