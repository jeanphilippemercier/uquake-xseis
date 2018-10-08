from importlib import reload
import numpy as np
import os
from io import BytesIO
# import struct
# import glob
# import datetime
from obspy import read
from spp.utils.kafka import KafkaHandler
import yaml

from xseis2 import xutil
from xseis2 import xflow
from xseis2 import xspy

nthreads = int(4)
debug = int(0)
dsr = float(3000.)
wlen_sec = float(1.0)

common_dir = os.environ['SPP_COMMON']
# tts_dir = os.path.join(common_dir, 'NLL/time/')
tts_dir = '/mnt/seismic_shared_storage/time/'

ttable, stalocs, namedict, gdef = xutil.ttable_from_nll_grids(tts_dir, key="OT.P")
# ttable, stalocs, namedict, gdef = xutil.ttable_from_nll_grids(tts_path, key="OT.S")
ttable = (ttable * dsr).astype(np.uint16)
ngrid = ttable.shape[1]
tt_ptrs = np.array([row.__array_interface__['data'][0] for row in ttable])


dsr = float(3000.)
ttP, slocs, ndict, gdef = xutil.ttable_from_nll_grids(tts_dir, key="OT.P")
ttP = (ttP * dsr).astype(np.uint16)
# ttS, slocs, ndict, gdef = xutil.ttable_from_nll_grids(tts_dir, key="OT.S")
# ttS = (ttS * dsr).astype(np.uint16)

config_dir = os.environ['SPP_CONFIG']
fname = os.path.join(config_dir, 'data_connector_config.yaml')

with open(fname, 'r') as cfg_file:
	params = yaml.load(cfg_file)

logdir = params['data_connector']['logging']['log_directory']
# Create Kafka Object
brokers = params['data_connector']['kafka']['brokers']
topic_in = params['data_connector']['kafka']['topic']
# consumer = KafkaHandler.consume_from_topic(topic_in, brokers)
consumer = KafkaHandler.consume_from_topic(topic_in, brokers, group_id='interloc_group')
topic_out = 'interloc'  # need to get this from a config file later


print("Awaiting Kafka mseed messsages")
for msg_in in consumer:
	print("Received Key:", msg_in.key)
	st = read(BytesIO(msg_in.value))
	# st = read(mseed_file)
	print(st)
	xflow.prep_stream(st)
	data, t0, stations, chanmap = xflow.build_input_data(st, wlen_sec, dsr)
	ikeep = np.array([namedict[k] for k in stations])
	npz_file = os.path.join(logdir, "iloc_" + str(t0) + ".npz")
	out = xspy.pySearchOnePhase(data, dsr, chanmap, stalocs[ikeep], tt_ptrs[ikeep],
								 ngrid, nthreads, debug, npz_file)
	vmax, imax, iot = out
	lmax = xutil.imax_to_xyz_gdef(imax, gdef)
	ot_epoch = (t0 + iot / dsr).datetime.timestamp()
	print("power: %.3f, ix_grid: %d, ix_ot: %d" % (vmax, imax, iot))
	print("utm_loc: ", lmax.astype(int))

	print("Sending Kafka interloc messsage")
	msg_out, key_out = xflow.encode_for_kafka(ot_epoch, lmax, vmax)
	kafka_handler_obj = KafkaHandler(brokers)
	kafka_handler_obj.send_to_kafka(topic_out, msg_out, key_out)
	kafka_handler_obj.producer.flush()
	print("==================================================================")

