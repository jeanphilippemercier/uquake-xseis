import os
from datetime import datetime, timedelta
import numpy as np
import itertools
# from sqlalchemy import Table, Column, Integer, String, MetaData, ForeignKey, DateTime, Float, Sequence, ARRAY, LargeBinary, Boolean
# from sqlalchemy.sql import select
# from sqlalchemy import create_engine
# from sqlalchemy.orm import sessionmaker
# from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import MetaData
from xseis2 import xutil
from xseis2 import xchange
# from xseis2 import xio

from xseis2.xsql import Channel, Station, StationPair, VelChange, XCorr, ChanPair

from loguru import logger

# name = Column(String(10), primary_key=True)
#     component = Column(String(5))
#     station = Column(String(10))
#     quality = Column(Integer)
#     samplerate = Column(Float)


def load_npz_continuous(fname):
    from obspy import UTCDateTime

    with np.load(fname) as npz:
        sr = float(npz['sr'])
        chan_names = npz['chans']
        starttime = UTCDateTime(str(npz['start_time'])).datetime
        data = npz['data'].astype(np.float32)
        data[data == 0] = -1
    endtime = starttime + timedelta(seconds=data.shape[1] / sr)

    return data, sr, starttime, endtime, chan_names


def ckeys_from_stapairs(pairs):

    db_corr_keys = []

    for pair in pairs:
        # print(pair)
        sta1, sta2 = pair.station1, pair.station2
        for chan1, chan2 in itertools.product(sta1.channels, sta2.channels):
            corr_key = f".{chan1}_.{chan2}"
            db_corr_keys.append(corr_key)

    return np.array(db_corr_keys)


def fill_tables_sta_chan(stations, session, clear=True):

    if clear:
        session.query(StationPair).delete()
        session.query(Channel).delete()
        session.query(Station).delete()
        session.commit()

    sta_rows = []
    chan_rows = []

    for sta in sorted(stations, key=lambda x: x.code):
        chans = sorted(sta.channels, key=lambda x: x.code)
        chan_names = []

        for chan in chans:
            chan_name = f"{sta.code}.{chan.code}"
            chan_rows.append(Channel(name=chan_name, component=chan.code, station_name=sta.code, quality=1))

            chan_names.append(chan_name)

        db_station = Station(name=sta.code, channels=chan_names, location=sta.loc)
        sta_rows.append(db_station)

    session.add_all(sta_rows)
    session.add_all(chan_rows)
    session.commit()


# def fill_table_stations(stations, session, clear=True):

#     if clear:
#         session.query(Station).delete()

#     rows = []
#     for i, sta in enumerate(stations):
#         chans = sorted([c.code for c in sta.channels])
#         db_station = Station(code=sta.code, channels=chans, location=sta.loc)
#         rows.append(db_station)

#     session.add_all(rows)
#     session.commit()


def fill_table_station_pairs(stations, session, clear=True):

    if clear:
        session.query(StationPair).delete()

    stations_sorted = sorted(stations, key=lambda x: x.name)

    rows = []
    for sta1, sta2 in itertools.combinations(stations_sorted, 2):
        inter_dist = xutil.dist(sta1.location, sta2.location)
        name = f"{sta1.name}_{sta2.name}"
        pair = StationPair(name=name, dist=inter_dist, station1_name=sta1.name, station2_name=sta2.name)
        rows.append(pair)

    session.add_all(rows)
    session.commit()


def fill_table_xcorrs(dc, nstack, ckeys, starttime, endtime, session, rhandle, nstack_min_percent=0):
    """
    Fill database with cross-correlations
    """
    ncc, nsamp = dc.shape
    logger.info(f'Adding {ncc} correlations for {starttime} to database')
    length_hours = (endtime - starttime).total_seconds() / 3600.

    pipe = rhandle.pipeline()
    rows = []
    for i, sig in enumerate(dc):
        if nstack[i] < nstack_min_percent:
            continue
        # print(i)
        ckey = ckeys[i]
        k1, k2 = ckey.split("_")
        c1, c2 = k1[1:], k2[1:]
        spair = f"{k1.split('.')[1]}_{k2.split('.')[1]}"
        dkey = f"{str(starttime)} {ckey}"

        pipe.set(dkey, xutil.array_to_bytes(sig))
        rows.append(XCorr(corr_key=ckey, channel1_name=c1, channel2_name=c2, stationpair_name=spair, start_time=starttime, length=length_hours, nstack=float(nstack[i]), waveform_redis_key=dkey))

    pipe.execute()  # add data to redis
    session.add_all(rows)  # add metadata rows to sql
    session.commit()


def xcorr_load_waveforms(xcorr_objects, rhandle):
    for xcorr in xcorr_objects:
        xcorr.waveform = xutil.bytes_to_array(rhandle.get(xcorr.waveform_redis_key))


def ckey_dists(ckeys, stations):

    sites = [station.code for station in stations]
    site_locs = [station.loc for station in stations]
    ldict = dict(zip(sites, site_locs))

    dists = dict()
    for ck in ckeys:
        c1, c2 = ck.split('_')
        dd = xutil.dist(ldict[c1[:-2]], ldict[c2[:-2]])
        dists[ck] = dd
    return dists


def sql_drop_tables(db):

    metadata = MetaData(db, reflect=True)

    for table in reversed(metadata.sorted_tables):
        table.drop(db)


def fill_table_chanpairs(stations, session, min_pair_dist=0, max_pair_dist=99999):

    logger.info(f'Clear and fill ChanPair PSQL table with {len(stations)} stations')
    session.query(ChanPair).delete()

    stations_sorted = sorted(stations, key=lambda x: x.code)

    rows = []
    for sta1, sta2 in itertools.combinations(stations_sorted, 2):
        inter_dist = xutil.dist(sta1.loc, sta2.loc)
        # print(sta1.code, sta2.code, inter_dist)
        if inter_dist < min_pair_dist or inter_dist > max_pair_dist:
            continue

        for chan1, chan2 in itertools.product(sta1.channels, sta2.channels):
            # print(chan1.code, chan2.code)
            corr_key = f".{sta1.code}.{chan1.code.upper()}_.{sta2.code}.{chan2.code.upper()}"
            # print(corr_key)
            rows.append(ChanPair(corr_key=corr_key, inter_dist=inter_dist,
                                 station1=sta1.code, station2=sta2.code))

    session.add_all(rows)  # add rows to sql
    session.commit()

    logger.info(f'Added {len(rows)} channel pairs')


def measure_dvv_xcorrs(session, rhandle):

    max_pair_dist = 1000
    coda_start_vel = 3200.
    sr = 1000.0
    coda_end_sec = 0.8
    wlen_sec = 0.05
    whiten_freqs = np.array([80, 100, 250, 300])
    nrecent = 10

    ckeys = np.unique(session.query(XCorr.ckey).all())
    cpairs = session.query(ChanPair).filter(ChanPair.ckey.in_(ckeys)).filter(ChanPair.dist < max_pair_dist).all()

    for cpair in cpairs:

        dist = cpair.dist
        ckey = cpair.ckey

        coda_start_sec = dist / coda_start_vel
        print(f"{ckey}: {dist:.2f}m")

        ccfs = session.query(XCorr).filter_by(ckey=ckey).order_by(XCorr.time.desc()).limit(nrecent).all()[::-1]

        for icc in range(1, len(ccfs)):
            # print(i)
            cc_ref = ccfs[icc - 1]
            cc_cur = ccfs[icc]
            sig_ref = xutil.bytes_to_array(rhandle.get(cc_ref.data))
            sig_cur = xutil.bytes_to_array(rhandle.get(cc_cur.data))
            dvv, error = xchange.dvv(sig_ref, sig_cur, sr, wlen_sec, whiten_freqs, coda_start_sec, coda_end_sec)
            cc_cur.dvv = dvv
            cc_cur.error = error


def measure_dvv_stations(session):
    nrecent = 10

    ckeys = np.unique(session.query(XCorr.ckey).all())

    stas = []
    for ck in ckeys:
        c1, c2 = ck.split('_')
        stas.extend([c1[:-2], c2[:-2]])

    stas = np.unique(stas)

    res = session.query(XCorr.time).distinct().all()
    times = np.sort([x[0] for x in res])[-nrecent:]

    for dtime in times:
        for sta in stas:
            out = session.query(XCorr.dvv).filter(XCorr.time == dtime).filter(XCorr.ckey.like(f"%{sta}%")).filter(XCorr.dvv.isnot(None)).all()
            if len(out) == 0:
                continue
            print(sta, len(out))
            session.add(VelChange(time=dtime, sta=sta, dvv=np.median(out), error=np.std(out)))
        session.commit()


def get_continuous_fake(start_time, end_time, stations, fband=[20, 400], sr=1000.0):
    from microquake.core.stream import Trace, Stream
    from obspy import UTCDateTime

    start_time = UTCDateTime(start_time)
    end_time = UTCDateTime(end_time)

    nsamp = int((end_time - start_time) * sr)
    traces = []
    for sta in stations:
        # loc = sta.loc
        dat = xutil.band_noise(fband, sr, nsamp)
        for chan in ['X', 'Y', 'Z']:
            # chans.append(f"{sta.code}.{chan.code}")
            # locs.append(loc)
            tr = Trace(data=dat)
            tr.stats.starttime = start_time
            tr.stats.sampling_rate = sr
            tr.stats.channel = chan
            tr.stats.station = sta
            traces.append(tr)

    st = Stream(traces=traces)
    return st




# def fill_table_xcorrs(hstream, session, rhandle):

#     # channels to correlate can either be (1) taken from db (2) or params
#     chans = hstream.channels
#     sr_raw = hstream.samplerate
#     sr = sr_raw

#     # PARAMS - config
#     whiten_freqs = np.array([60, 80, 320, 350])
#     cclen = 20.0
#     keeplag = 1.0
#     stepsize = cclen
#     onebit = True
#     stacklen = 3600.0
#     # stacklen = 1000

#     # Params to function
#     times = [hstream.starttime + timedelta(seconds=i * stacklen) for i in range(5)]

#     for t0 in times:
#         print(t0)
#         t1 = t0 + timedelta(seconds=stacklen)

#         # python generator which yields slices of data
#         datgen = hstream.slice_gen(t0, t1, chans, cclen, stepsize=stepsize)

#         dc, ckeys_ix = xchange.xcorr_stack_slices(
#             datgen, chans, cclen, sr_raw, sr, keeplag, whiten_freqs, onebit=onebit)
#         ckeys = [f"{ck[0]}_{ck[1]}" for ck in chans[ckeys_ix]]

#         pipe = rhandle.pipeline()
#         rows = []
#         for i, sig in enumerate(dc):
#             # print(i)
#             ckey = ckeys[i]
#             dkey = f"{str(t0)} {ckey}"
#             pipe.set(dkey, xutil.array_to_bytes(sig))
#             rows.append(XCorr(time=t0, ckey=ckey, data=dkey))

#         pipe.execute()  # add data to redis
#         session.add_all(rows)  # add rows to sql
#         session.commit()

# def fill_table_chanpairs(stations, session):

#     chans = []

#     for sta in stations:
#         for chan in sta.channels:
#             chans.append(f".{sta.code}.{chan.code.upper()}")
#     chans = np.array(chans)

#     ckeys = xutil.unique_pairs(chans)
#     ckeys = xutil.ckeys_remove_intersta_str(ckeys)
#     ckeys = np.array([f"{ck[0]}_{ck[1]}" for ck in ckeys])

#     sites = [station.code for station in stations]
#     site_locs = [station.loc for station in stations]
#     ldict = dict(zip(sites, site_locs))

#     rows = []
#     # dists = dict()
#     for ck in ckeys:
#         c1, c2 = ck.split('_')
#         sta1, sta2 = c1[:-2], c1[:-2]
#         dd = xutil.dist(ldict[sta1], ldict[sta2])
#         rows.append(ChanPair(corr_key=ck, inter_dist=dd, station1=sta1, station2=sta2))
#         # dists[ck] = dd

#     # for k, v in dists.items():
#     #     rows.append(ChanPair(corr_key=k, dist_meters=v))
#     #     # print(k, v)

#     session.add_all(rows)  # add rows to sql
#     session.commit()
