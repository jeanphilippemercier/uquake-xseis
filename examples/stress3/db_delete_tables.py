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

metadata = MetaData(db)
metadata.reflect()

for table in reversed(metadata.sorted_tables):
    # db.execute(tbl.delete())
    table.drop(db)
    # session.execute(tbl.drop())
    session.commit()

Base.metadata.create_all(db)
session.commit()


##################################
rh = redis.Redis(host='localhost', port=6379, db=0)
rh.flushall()
###################################
