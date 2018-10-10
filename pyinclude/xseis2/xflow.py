# -*- coding: utf-8 -*-
# @Author: Philippe Dales
# @Date:   2018-10-02 13:16:23
# @Last Modified by:   Philippe Dales
# @Last Modified time: 2018-10-10 15:00:26
import numpy as np
import struct
from xseis2 import xutil


def stream_to_array(st, t0, npts_fix, taplen=50):
	sr = st[0].stats.sampling_rate
	nsig = len(st)
	data = np.zeros((nsig, npts_fix), dtype=np.float32)
	for i, tr in enumerate(st):
		i0 = int((tr.stats.starttime - t0) * sr + 0.5)
		sig = tr.data - np.mean(tr.data)
		xutil.taper_data(sig, taplen)
		slen = min(len(sig), npts_fix - i0)
		data[i, i0: i0 + slen] = sig[:slen]
	return data


def prep_stream(st):
	for tr in st:
		tr.stats.station = tr.stats.station.zfill(3)
	st.sort()
	# sr = st[0].stats.sampling_rate
	# st.detrend()
	# if sr > dsr:
	# 	st.decimate(int(sr / dsr))


def build_input_data(st, wlen_sec, dsr=None):
	t0 = np.min([tr.stats.starttime for tr in st])
	sr = st[0].stats.sampling_rate
	data = stream_to_array(st, t0, npts_fix=int(wlen_sec * sr))
	# data -= data.mean(axis=1, keepdims=True)
	# data = data - data.mean(axis=1, keepdims=True)
	if dsr is not None and sr > dsr:
		data = xutil.decimate(data, sr, int(sr / dsr))
	stations = np.array([tr.stats.station for tr in st])
	unique = np.unique(stations)
	unique_dict = dict(zip(unique, np.arange(len(unique))))
	chanmap = np.array([unique_dict[chan] for chan in stations], dtype=np.uint16)
	return data, t0, unique, chanmap


def encode_for_kafka(ot_epoch, loc, power):
	msg = np.array([ot_epoch, loc[0], loc[1], loc[2], power], dtype=np.float64)
	kaf_msg = struct.pack('%sd' % len(msg), *msg)
	kaf_key = str(ot_epoch).encode('utf-8')
	kaf_key = ("iloc_%d" % (ot_epoch)).encode('utf-8')

	return kaf_msg, kaf_key


def parse_event_xml(fle):
	import xml.etree.ElementTree as ET
	tree = ET.parse(fle)
	root = tree.getroot()
	vals = root[0][0]
	loc = np.zeros(3)
	loc[0] = vals.find("{MICROQUAKE}LOCATION_X").text
	loc[1] = vals.find("{MICROQUAKE}LOCATION_Y").text
	loc[2] = vals.find("{MICROQUAKE}LOCATION_Z").text

	return loc


def split_mseed(st, wlen):
	from obspy import Stream, Trace
	npts = len(st[0].data)
	nchunks = int(npts / wlen)
	for i in range(nchunks):
		stnew = Stream()
		# stmp = st.copy()
		for tr in st:
			trnew = Trace()
			trnew.stats = tr.stats

	

