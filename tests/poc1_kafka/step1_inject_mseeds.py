import numpy as np
import os
# import struct
import glob
from importlib import reload
# import datetime
from obspy import read
from xseis import xutil
from xseis import xloc
from io import BytesIO
from spp.utils.kafka import KafkaHandler
import yaml
import time

config_dir = os.environ['SPP_CONFIG']
fname = os.path.join(config_dir, 'data_connector_config.yaml')

with open(fname, 'r') as cfg_file:
	params = yaml.load(cfg_file)
# Create Kafka Object
brokers = params['data_connector']['kafka']['brokers']
topic = params['data_connector']['kafka']['topic']
kaf_handle = KafkaHandler(brokers)

data_src = "/mnt/seismic_shared_storage/OT_seismic_data/"
# data_src = params['data_connector']['data_source']['location']
MSEEDS = np.sort(glob.glob(data_src + '*.mseed'))

print("Sending kafka mseed messsages")

for fle in MSEEDS[-10:]:
	fsizemb = os.path.getsize(fle) / 1024.**2
	print("%s | %.2f mb" % (os.path.basename(fle), fsizemb))
	if (fsizemb < 3 or fsizemb > 15):
		print("skipping file")
		continue

	st = read(fle)  # could read fle directly into bytes io
	buf = BytesIO()
	st.write(buf, format='MSEED')
	msg_out = buf.getvalue()
	key_out = str(st[0].stats.starttime).encode('utf-8')
	kaf_handle.send_to_kafka(topic, msg_out, key_out)
	kaf_handle.producer.flush()
	print("==================================================================")
