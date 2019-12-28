from importlib import reload
import os
import numpy as np
from glob import glob
from datetime import datetime, timedelta
import itertools
import time

import redis
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xseis2 import xplot
from xseis2 import xutil
from xseis2 import xchange
from xseis2 import xchange_workflow as flow
from xseis2.xsql import Base, Station, XCorr, ChanPair, StationDvv
# from xseis2 import xsql
from pytz import utc

from loguru import logger
import matplotlib.pyplot as plt

from microquake.core.settings import settings

plt.ion()

#######################################################
#  xcorr processing params
#######################################################
rolling_keep_hours = settings.COMPUTE_XCORRS.rolling_database_keep_hours
# rolling_keep_hours = 24 * 3

#######################################################
#  connect to databases
#######################################################

logger.info('Connect to psql database')
db_string = "postgres://postgres:postgres@localhost"
db = create_engine(db_string)
print(db.table_names())
session = sessionmaker(bind=db)()
Base.metadata.create_all(db)

logger.info('Connect to redis database')
rhandle = redis.Redis(host='localhost', port=6379, db=0)

#######################################################
#  delete old entries
#######################################################

ckeys_db = np.array(session.query(XCorr.corr_key).distinct().order_by(XCorr.corr_key.asc()).all()).flatten()
logger.info(f'{len(ckeys_db)} unique xcorr pairs in db')
reload(flow)

ckey = ckeys_db[0]
xcorrs = session.query(XCorr).filter_by(corr_key=ckey).order_by(XCorr.start_time.desc()).all()[::-1]

######################################
# curtime = datetime.utcnow()
curr_time = xcorrs[-1].start_time
curr_time.replace(tzinfo=utc)
cutoff_time = curr_time - timedelta(hours=rolling_keep_hours)


xcorrs = session.query(XCorr).filter(XCorr.start_time < cutoff_time).all()
logger.info(f'Deleting {len(xcorrs)} entries from psql and redis')

if len(xcorrs) > 0:
    redis_keys = [xc.waveform_redis_key for xc in xcorrs]
    rhandle.delete(*redis_keys)

    delete_q = XCorr.__table__.delete().where(XCorr.start_time < cutoff_time)
    session.execute(delete_q)
    session.commit()
