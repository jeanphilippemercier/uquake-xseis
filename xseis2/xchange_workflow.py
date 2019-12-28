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

from xseis2.xsql import Channel, Station, StationPair, StationDvv, XCorr, ChanPair

from loguru import logger

# name = Column(String(10), primary_key=True)
#     component = Column(String(5))
#     station = Column(String(10))
#     quality = Column(Integer)
#     samplerate = Column(Float)


def unique_ordered(session, sql_class_attr):
    return np.array(session.query(sql_class_attr).distinct().order_by(sql_class_attr.asc()).all()).flatten()


def time_group_and_stack_xcorrs(xcorrs, nhour_stack):

    xgroups, dbins = group_xcorrs_by_time(xcorrs, nhour_stack)
    # [len(x) for x in xgroups]

    xcorrs_stack = []
    for xg, new_start_time in zip(xgroups, dbins):
        xnew = stack_xcorrs(xg, new_start_time, nhour_stack)
        xcorrs_stack.append(xnew)

    return xcorrs_stack


def stack_xcorrs(xcorrs, new_start_time, new_length):
    ex = xcorrs[0]
    # pipe.set(dkey, xutil.array_to_bytes(sig))

    sig = np.mean(np.array([x.waveform for x in xcorrs]), axis=0)
    # new_nstack = np.mean(np.array([x.nstack for x in xcorrs]))
    nhour_data = np.sum(np.array([x.nstack / 100 * x.length for x in xcorrs]))
    new_nstack = nhour_data / new_length * 100

    xnew = XCorr(corr_key=ex.corr_key, start_time=new_start_time, length=new_length, nstack=new_nstack, samplerate=ex.samplerate)
    # xnew = XCorr(corr_key=ex.corr_key, channel1_name=ex.channel1_name, channel2_name=ex.channel2_name, stationpair_name=ex.stationpair_name, start_time=new_start_time, length=new_length, nstack=new_nstack, samplerate=ex.samplerate)
    xnew.waveform = sig

    return xnew


def write_xcorrs(xcorrs, session, rhandle):

    pipe = rhandle.pipeline()
    for xc in xcorrs:
        redis_key = f"{str(xc.start_time)} {xc.corr_key}"
        xc.waveform_redis_key = redis_key
        pipe.set(redis_key, xutil.array_to_bytes(xc.waveform))

    pipe.execute()  # add data to redis
    session.add_all(xcorrs)  # add metadata rows to sql
    session.commit()


def group_xcorrs_by_time(xcorrs, blocklen_hours):

    wlen = timedelta(hours=blocklen_hours)

    dates = np.array([x.start_time for x in xcorrs])
    start = xutil.hour_round_down(np.min(dates))
    stop = xutil.hour_round_up(np.max(dates))
    dbins = xutil.datetime_bins(start, stop, wlen)

    xgroups = [[] for i in range(len(dbins))]

    for xc in xcorrs:
        ind = np.argmin(np.abs(dbins - xc.start_time))
        xgroups[ind].append(xc)

    xgroups = np.array(xgroups)
    ikeep = np.where(np.array([len(x) for x in xgroups]) > 0)[0]

    return xgroups[ikeep], dbins[ikeep]


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


def load_npz_continuous_meta(fname):
    from obspy import UTCDateTime

    with np.load(fname) as npz:
        sr = float(npz['sr'])
        chan_names = npz['chans']
        starttime = UTCDateTime(str(npz['start_time'])).datetime
        # data = npz['data'].astype(np.float32)
        # data[data == 0] = -1
    endtime = starttime + timedelta(minutes=10)
    data = None
    return data, sr, starttime, endtime, chan_names


def ckeys_from_stapairs(pairs):

    db_corr_keys = []

    for pair in pairs:
        # print(pair)
        sta1, sta2 = pair.station1, pair.station2
        for comp1, comp2 in itertools.product(sta1.channels, sta2.channels):
            chan1 = f".{sta1.name}.{comp1.upper()}"
            chan2 = f".{sta2.name}.{comp2.upper()}"
            corr_key = f"{chan1}_{chan2}"
            db_corr_keys.append(corr_key)

    return np.array(db_corr_keys)


# def fill_tables_sta_chan(stations, session, clear=True):

#     if clear:
#         session.query(StationPair).delete()
#         session.query(Channel).delete()
#         session.query(Station).delete()
#         session.commit()

#     sta_rows = []
#     chan_rows = []

#     for sta in sorted(stations, key=lambda x: x.code):
#         chans = sorted(sta.channels, key=lambda x: x.code)
#         chan_names = []

#         for chan in chans:
#             chan_name = f"{sta.code}.{chan.code}"
#             chan_rows.append(Channel(name=chan_name, component=chan.code, station_name=sta.code, quality=1))

#             chan_names.append(chan_name)

#         db_station = Station(name=sta.code, channels=chan_names, location=sta.loc)
#         sta_rows.append(db_station)

#     session.add_all(sta_rows)
#     session.add_all(chan_rows)
#     session.commit()


def fill_table_stations(stations, session, clear=True):

    if clear:
        session.query(Station).delete()

    rows = []

    for sta in sorted(stations, key=lambda x: x.code):
        chans = sorted([c.code for c in sta.channels])
        db_station = Station(name=sta.code, channels=chans, location=sta.loc)
        rows.append(db_station)

    session.add_all(rows)
    session.commit()


def fill_table_station_pairs(session, stations=None, clear=True):
    if stations is None:
        stations = session.query(Station).all()

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


def fill_table_xcorrs(dc, sr, nstack, ckeys, starttime, endtime, session, rhandle, nstack_min_percent=0):
    """
    Fill database with cross-correlations
    """
    ncc, nsamp = dc.shape
    logger.info(f'Adding {ncc} correlations for {starttime} to database')
    length_hours = (endtime - starttime).total_seconds() / 3600.
    # length_minutes = int((endtime - starttime).total_seconds() / 60)

    pipe = rhandle.pipeline()
    rows = []
    for i, sig in enumerate(dc):
        if nstack[i] < nstack_min_percent:
            continue
        # print(i)
        ckey = ckeys[i]
        # k1, k2 = ckey.split("_")
        # c1, c2 = k1[1:], k2[1:]
        # spair = f"{k1.split('.')[1]}_{k2.split('.')[1]}"
        dkey = f"{str(starttime)} {ckey}"

        pipe.set(dkey, xutil.array_to_bytes(sig))
        rows.append(XCorr(corr_key=ckey, start_time=starttime, length=length_hours, nstack=float(nstack[i]), waveform_redis_key=dkey, samplerate=sr, stacked=False))
        # rows.append(XCorr(corr_key=ckey, channel1_name=c1, channel2_name=c2, stationpair_name=spair, start_time=starttime, length=length_hours, nstack=float(nstack[i]), waveform_redis_key=dkey, samplerate=sr))

    pipe.execute()  # add data to redis
    session.add_all(rows)  # add metadata rows to sql
    session.commit()


def fill_table_chanpairs(session, pair_dist_min=0, pair_dist_max=99999, bad_chans=None):

    logger.info(f'Clear and fill ChanPair PSQL table')
    session.query(ChanPair).delete()

    sta_pairs = session.query(StationPair).filter(StationPair.dist.between(pair_dist_min, pair_dist_max)).filter().all()

    rows = []

    for pair in sta_pairs:
        sta1, sta2 = pair.station1, pair.station2
        dist = pair.dist
        for comp1, comp2 in itertools.product(sta1.channels, sta2.channels):
            chan1 = f".{sta1.name}.{comp1.upper()}"
            chan2 = f".{sta2.name}.{comp2.upper()}"
            corr_key = f"{chan1}_{chan2}"
            if chan1 in bad_chans or chan2 in bad_chans:
                # print(f"bad {corr_key}")
                continue

            # print(corr_key)
            rows.append(ChanPair(name=corr_key, dist=dist, stationpair_name=pair.name))

    session.add_all(rows)  # add rows to sql
    session.commit()

    logger.info(f'Added {len(rows)} channel pairs')


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


def measure_dvv_xcorrs(session, rhandle):

    pair_dist_max = 1000
    coda_start_vel = 3200.
    sr = 1000.0
    coda_end_sec = 0.8
    wlen_sec = 0.05
    whiten_freqs = np.array([80, 100, 250, 300])
    nrecent = 10

    ckeys = np.unique(session.query(XCorr.ckey).all())
    cpairs = session.query(ChanPair).filter(ChanPair.ckey.in_(ckeys)).filter(ChanPair.dist < pair_dist_max).all()

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


def measure_dvv_stations(session, nhour_stack):

    ckeys = unique_ordered(session, XCorr.corr_key)

    stas = []
    for ck in ckeys:
        # stas.extend([c[:-1] for c in ck.split('_')])
        stas.extend([c.split('.')[1] for c in ck.split('_')])

    stas = np.unique(stas)
    # times = np.array(session.query(XCorr.start_time).distinct().all()).flatten()
    times = np.array(session.query(XCorr.start_time).filter(XCorr.length == nhour_stack).distinct().all()).flatten()

    for dtime in times:
        for sta in stas:
            out = session.query(XCorr.velocity_change).filter(XCorr.start_time == dtime).filter(XCorr.corr_key.like(f"%.{sta}.%")).filter(XCorr.velocity_change.isnot(None)).all()

            if len(out) == 0:
                continue
            print(sta, len(out))
            session.add(StationDvv(start_time=dtime, station=sta, velocity_change=np.median(out), error=np.std(out), navg=len(out)))
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
