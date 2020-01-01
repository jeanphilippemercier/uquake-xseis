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

from loguru import logger
import matplotlib.pyplot as plt

from microquake.core.settings import settings

plt.ion()

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
#  combine multiple
#######################################################

nhour_stack = settings.COMPUTE_XCORRS.stack_length_hours
print(f"nhour_stack: {nhour_stack}")

ckeys_db = np.array(session.query(XCorr.corr_key).filter_by(status=0).distinct().order_by(XCorr.corr_key.asc()).all()).flatten()
logger.info(f'{len(ckeys_db)} pairs with unprocessed chunks')
reload(flow)

# # delete already stacked corrs
# delete_q = XCorr.__table__.delete().where(XCorr.stacked == True)
# session.execute(delete_q)
# session.commit()

for i, ckey in enumerate(ckeys_db):
    logger.info(f"Processing {ckey} {i} / {len(ckeys_db)}")

    xcorrs = session.query(XCorr).filter_by(corr_key=ckey).filter_by(status=0).order_by(XCorr.start_time.asc()).all()
    if len(xcorrs) == 0:
        continue

    flow.xcorr_load_waveforms(xcorrs, rhandle)
    xcorrs_stack = flow.time_group_and_stack_xcorrs(xcorrs, nhour_stack)
    flow.write_xcorrs(xcorrs_stack, session, rhandle)


#######################################################
#  measure dvv
#######################################################
reload(flow)
params = settings.COMPUTE_VELCHANGE
print(params)

logger.info(f'Measuring dvv xcorrs')
flow.measure_dvv_xcorrs(session, rhandle, params)

logger.info(f'Measuring dvv stations')
flow.measure_dvv_stations(session)


xcorrs = session.query(XCorr).filter_by(status=2).all()
logger.info(f'{len(xcorrs)} entries from psql and redis')


#######################################################
#  delete processed chunks (status=1)
#######################################################
status = 2
# xcorrs = session.query(XCorr).filter_by(status=1).all()
xcorrs = session.query(XCorr).filter_by(status=status).all()
logger.info(f'Deleting {len(xcorrs)} entries from psql and redis')

if len(xcorrs) > 0:
    redis_keys = [xc.waveform_redis_key for xc in xcorrs]
    rhandle.delete(*redis_keys)

    delete_q = XCorr.__table__.delete().where(XCorr.status == status)
    session.execute(delete_q)
    session.commit()


#######################################################
#  reprocess chunks
#######################################################

xcorrs = session.query(XCorr).filter_by(status=1).all()
logger.info(f'{len(xcorrs)} entries from psql and redis')

if len(xcorrs) > 0:
    for xc in xcorrs:
        xc.status = 0
    session.commit()
