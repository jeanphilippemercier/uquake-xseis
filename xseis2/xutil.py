""" utils."""

import numpy as np
import math
from scipy.fftpack import fft, ifft, fftfreq
from scipy.signal import sosfilt, zpk2sos, iirfilter
import os
import glob
import itertools
import io
from scipy.stats import linregress
from datetime import timedelta, datetime
from scipy.signal import hilbert
from scipy.ndimage.filters import gaussian_filter


def hour_round_nearest(t):
    # Rounds to nearest hour by adding a timedelta hour if minute >= 30
    return (t.replace(second=0, microsecond=0, minute=0, hour=t.hour) + timedelta(hours=t.minute // 30))


def hour_round_down(t):
    return t.replace(second=0, microsecond=0, minute=0, hour=t.hour)


def hour_round_up(t):
    return t.replace(second=0, microsecond=0, minute=0, hour=t.hour) + timedelta(hours=1)


def datetime_bins(start, stop, wlen, stepsize=None):

    if stepsize is None:
        stepsize = wlen

    bins = []
    curr = start
    while curr < stop:
        d0 = curr
        d1 = curr + wlen
        # bins.append([d0, d1])
        bins.append(d0)
        curr += stepsize

    return np.array(bins)


def ckeys_remove_chans(ckeys, names):
    ikeep = []
    for i, ck in enumerate(ckeys):
        c1, c2 = ck.split('_')
        if c1 in names or c2 in names:
            continue
        ikeep.append(i)

    return ckeys[np.array(ikeep)]


def ckey_to_chan_stats(ckeys, vals):

    d = dict()
    for i, ck in enumerate(ckeys):
        c1, c2 = ck.split('_')
        if c1 not in d:
            d[c1] = [vals[i]]
            continue
        if c2 not in d:
            d[c2] = [vals[i]]
            continue

        d[c1].append(vals[i])
        d[c2].append(vals[i])

    return np.array(list(d.keys())), np.array(list(d.values()))


def linear_detrend_nan(y):
    x = np.arange(len(y))
    not_nan_ind = ~np.isnan(y)
    m, b, r_val, p_val, std_err = linregress(x[not_nan_ind], y[not_nan_ind])
    detrend_y = y - (m * x + b)
    return detrend_y


def roundup(x, nearest_multiple):
    return int(math.ceil(x / nearest_multiple)) * int(nearest_multiple)


def whiten_sig(sig, sr, whiten_freqs, pad_multiple=None):
    if pad_multiple is None:
        npad = len(sig)
    else:
        npad = roundup(len(sig), pad_multiple)

    whiten_win = freq_window(whiten_freqs, npad, sr)
    fsig = np.fft.rfft(sig, n=npad)
    fsig = whiten_win * phase(fsig)
    return np.fft.irfft(fsig)[:len(sig)]


def symmetric(data):
    """Create symmetric signal."""
    if data.ndim == 1:
        split = split_causals(data)
        split[0][0] = 0
        sym = np.sum(split, axis=0) / 2.
    elif data.ndim == 2:
        sym = (acaus2d(data) + caus2d(data)) / 2

    return sym


def split_causals(sig, overlap=0):
    """Split signals into causal/acausal."""
    length = len(sig)
    lh = length // 2
    pos_start = lh - overlap
    neg_end = lh + overlap

    if length % 2 != 0:
        neg_end += 1

    return np.array([sig[pos_start:], sig[:neg_end][::-1]])


def acaus2d(data):
    return np.fliplr(data[:, :data.shape[1] // 2])


def caus2d(data):
    return data[:, data.shape[1] // 2:]


def average_adjacent_rows(dat, nrow_avg):
    nrow, ncol = dat.shape
    nrow_new = nrow // nrow_avg
    row_mutliple = int(nrow_new * nrow_avg)

    out = dat[:row_mutliple].reshape((nrow_new, nrow_avg, ncol))
    out = np.mean(out, axis=1)
    return out


def to_dict_of_lists(list_of_dicts):
    ex_dict = list_of_dicts[0]
    dict_of_lists = {k: [] for k in ex_dict.keys()}

    for d in list_of_dicts:
        for k, v in d.items():
            dict_of_lists[k].append(v)

    for k, v in dict_of_lists.items():
        dict_of_lists[k] = np.array(v)

    return dict_of_lists


def maxnorm(dat, scale=1):
    out = dat - np.mean(dat)
    return out * scale / np.max(np.abs(out))


def xcorr_lagtimes(nsamp, sr=1):
    hl = nsamp / (2 * sr)
    return np.linspace(-hl, hl, nsamp)


def check_onebit_bool_conversion(sig_raw, set_zero_random=False):

    sig = sig_raw.copy()

    # to bool
    if set_zero_random is True:
        izero = np.where(sig == 0)[0]
        sig[izero] = np.random.choice([-1, 1], len(izero))
    sig[sig < 0] = 0
    sig = np.sign(sig).astype(np.bool_)

    # from bool
    sig = sig.astype(np.float32)
    sig[sig == 0] = -1

    return sig


def index_ckeys_split(corr_keys, chan_names):
    cdict = dict(zip(chan_names, np.arange(len(chan_names))))
    ckeys_ix = []
    for key in corr_keys:
        k0, k1 = key.split('_')
        ckeys_ix.append([cdict[k0], cdict[k1]])

    ckeys_ix = np.array(ckeys_ix)
    return ckeys_ix


def array_to_bytes(dat):
    output = io.BytesIO()
    np.save(output, dat)
    return output.getvalue()


def bytes_to_array(buf):
    return np.load(io.BytesIO(buf))


def acausal(data):
    return np.fliplr(data[:, :data.shape[1] // 2])


def causal(data):
    return data[:, data.shape[1] // 2:]


def ricker(freq, sr, length=0.2):
    f = freq
    dt = 1. / sr
    t = np.arange(-length / 2, (length - dt) / 2, dt)
    y = (1.0 - 2.0*(np.pi**2)*(f**2)*(t**2)) * np.exp(-(np.pi**2)*(f**2)*(t**2))
    return t, y


# def sizemb(nfloat):
#     size = nfloat * 4.0 / 1024**2
#     print("%d float = %.2f mb" % (nfloat, size))


def sizeof(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi']:
        if abs(num) < 1024.0:
            # return "%3.1f%s%s" % (num, unit, suffix)
            return [num, unit + suffix]
        num /= 1024.0


def ttsamp(dist, vel, sr):
    return int(dist / vel * sr + 0.5)


def integrate(sig):
    from scipy.integrate import cumtrapz
    return cumtrapz(sig, initial=0)


def attenuate(sig, sr, dist, Q, vel, gspread=True):

    npts = len(sig)
    fsig = fft(sig)
    freqs = fftfreq(npts, d=1. / sr)
    tstar = dist / (vel * Q)
    factor = np.exp(-np.pi * np.abs(freqs) * tstar)
    fsig *= factor
    sig = np.real(ifft(fsig))
    if gspread:
        sig /= dist
    return sig


def cc_avg_coeff(ccs, ckeys):
    N = np.unique(ckeys).shape[0]
    vals = np.zeros(N)
    count = np.zeros(N)

    for i, k in enumerate(ckeys):
        mx = np.max(ccs[i])
        vals[k] += mx
        count[k] += 1

    return vals / count


def get_pt(index, shape, spacing, origin):
    nx, ny, nz = shape
    iz = index % nz
    iy = ((index - iz) // nz) % ny
    ix = index // (nz * ny)

    loc = np.array([ix, iy, iz]) * spacing + origin
    return loc


def imax_to_xyz_gdef(index, gdef):
    shape, origin, spacing = gdef[:3], gdef[3:6], float(gdef[6])
    nx, ny, nz = shape
    iz = index % nz
    iy = ((index - iz) // nz) % ny
    ix = index // (nz * ny)

    loc = np.array([ix, iy, iz], dtype=float) * spacing + origin
    return loc


def gdef_to_points(gdef):

    shape, origin, spacing = gdef[:3], gdef[3:6], float(gdef[6])
    # nx, ny, nz = shape
    maxes = origin + shape * spacing
    x = np.arange(origin[0], maxes[0], spacing).astype(np.float32)
    y = np.arange(origin[1], maxes[1], spacing).astype(np.float32)
    z = np.arange(origin[2], maxes[2], spacing).astype(np.float32)
    points = np.zeros((np.product(shape), 3), dtype=np.float32)
    # points = np.stack(np.meshgrid(x, y, z), 3).reshape(3, -1).astype(np.float32)
    ix = 0
    for xv in x:
        for yv in y:
            for zv in z:
                points[ix] = [xv, yv, zv]
                ix += 1
    return points


def points2d(x, y):
    return np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])


def points3d(x, y, z):
    size = len(x) * len(y) * len(z)
    points = np.zeros((size, 3), dtype=np.float32)
    ix = 0
    for xv in x:
        for yv in y:
            for zv in z:
                points[ix] = [xv, yv, zv]
                ix += 1
    return points


def ttable_from_nll_grids(path, key="OT.P"):
    fles = np.sort(glob.glob(os.path.join(path, key + '*.time.buf')))
    hfles = np.sort(glob.glob(os.path.join(path, key + '*.time.hdr')))
    assert(len(fles) == len(hfles))
    stas = np.array([f.split('.')[-3].zfill(3) for f in fles], dtype='S4')
    isort = np.argsort(stas)
    fles = fles[isort]
    hfles = hfles[isort]
    names = stas[isort]

    vals = [read_nll_header(fle) for fle in hfles]
    sloc, shape, org, spacing = vals[0]
    slocs = np.array([v[0] for v in vals], dtype=np.float32)
    ngrid = np.product(shape)

    nsta = len(fles)
    tts = np.zeros((nsta, ngrid), dtype=np.float32)

    for i in range(nsta):
        tts[i] = np.fromfile(fles[i], dtype='f4')

    gdef = np.concatenate((shape, org, [spacing])).astype(np.int32)

    ndict = {}
    for i, sk in enumerate(names):
        ndict[sk.decode('utf-8')] = i

    return tts, slocs, ndict, gdef


def read_nll_header(fle):
    # print(fle)
    dat = open(fle).read().split()
    shape = np.array(dat[:3], dtype=int)
    org = np.array(dat[3:6], dtype=np.float32) * 1000.
    spacing = (np.array(dat[6:9], dtype=np.float32) * 1000.)[0]
    sloc = np.array(dat[12:15], dtype=np.float32) * 1000.

    return sloc, shape, org, spacing


def g2mad(grid):
    # return max, median and median absolute dev
    med = np.median(grid)
    mad = np.median(np.abs(grid - med))
    # mad = max(mad, 1)
    out = (grid - med) / mad
    return out


def randomize_freq_band(sig, band, sr):

    fsig = np.fft.rfft(sig)
    freqmin, freqmax = band
    freqs = np.fft.rfftfreq(len(sig), 1.0 / sr)
    idx = np.where(np.logical_and(freqs >= freqmin, freqs <= freqmax))[0]

    part = fsig[idx]
    amps = np.abs(part)
    rand = np.random.uniform(-np.pi, np.pi, len(part))
    angs = np.exp(1j * rand)
    fsig[idx] = amps * angs

    return np.fft.irfft(fsig)


# def randomize_band_freq(fsig, band, sr):

#     fnew = fsig.copy()
#     freqmin, freqmax = band
#     samples = len(fnew)
#     freqs = np.abs(fftfreq(samples, 1. / sr))
#     idx = np.where(np.logical_and(freqs >= freqmin, freqs <= freqmax))[0]

#     part = fnew[idx]
#     amps = np.abs(part)
#     rand = np.random.uniform(-np.pi, np.pi, len(part))
#     angs = np.exp(1j * rand)
#     fnew[idx] = amps * angs

#     return fnew


def roll_data(data, tts):
    droll = np.zeros_like(data)

    for i, sig in enumerate(data):
        droll[i] = np.roll(sig, -tts[i])
    return droll


def velstack(data, dists2src, sr, vels):

    dnorm = norm2d(data)
    dstack = np.zeros((len(vels), dnorm.shape[1]), dtype=np.float32)
    for ivel, vel in enumerate(vels):
        shifts = (dists2src / vel * sr + 0.5).astype(int)
        for i, shift in enumerate(shifts):
            dstack[ivel] += np.roll(dnorm[i], -shift)
    return dstack


def chan_groups(chanmap):
    return [np.where(sk == chanmap)[0] for sk in np.unique(chanmap)]


def comb_channels(data, cmap):

    groups = [np.where(sk == cmap)[0] for sk in np.unique(cmap)]
    dstack = np.zeros((len(groups), data.shape[1]))

    for i, grp in enumerate(groups):
        dstack[i] = np.mean(np.abs(data[grp]), axis=0)

    return dstack


def mlab_coords(locs, lims, spacing):
    return (locs - lims[:, 0]).T / spacing


def SearchClusters(data, dmin, zmin=0):

    inds = np.arange(data.shape[1])

    slocs = []
    vals = []
    for j, ix in enumerate(inds):
        print(j)
        lmax = data[:-1, ix, 1:]
        tmp = []
        for k, centroid in enumerate(lmax):
            if centroid[2] < zmin:
                tmp.append(0)
                continue
            diff = np.linalg.norm(lmax - centroid, axis=-1)
            tmp.append(np.where(diff < dmin)[0].size)

        imax = np.argmax(tmp)
        vals.append(tmp[imax])
        slocs.append(lmax[imax])

    vals = np.array(vals)
    slocs = np.array(slocs)

    return vals, slocs


def MeanDist(data):

    inds = np.arange(data.shape[1])
    vals = []
    for j, ix in enumerate(inds):
        print(j)
        lmax = data[:-1, ix, 1:]
        x, y, z = lmax.T
        err = np.std(x) + np.std(y) + np.std(z)
        vals.append(err)

    vals = np.array(vals) / 3

    return vals


def shift_locs(locs, unshift=False, vals=np.array([1.79236297e+05, 7.09943400e+06, 2.49199997e+02])):
    vals = np.array(vals)
    locs[:, 2] *= -1
    if unshift is True:
        return locs + vals
    else:
        return locs - vals


def shift_locs_ot(locs, unshift=False, vals=np.array([650000., 4766000., 0]), zdepth=1200.):
    vals = np.array(vals)
    lnew = locs.copy()

    if unshift is True:
        # lnew[:, 2] *= -1
        lnew.T[2] = (lnew.T[2] - zdepth) * -1
        return lnew + vals
    else:
        lnew.T[2] = zdepth - lnew.T[2]
        return lnew - vals

# def shift_locs_ot(locs, unshift=False, vals=np.array([650000., 4766000., 0])):
#   vals = np.array(vals)
#   if unshift is True:
#       return locs + vals
#   else:
#       return locs - vals


def pearson_coeff(sig1, sig2):
    return np.dot(sig1, sig2) / np.sqrt(energy(sig1) * energy(sig2))


def normVec(v):
    return v / norm(v)


def norm(v):
    return math.sqrt(np.dot(v, v))


def angle_between_vecs(v1, v2):
    return math.acos(np.dot(v1, v2) / (norm(v1) * norm(v2)))


def dist(l1, l2):
    return norm(np.array(l1) - np.array(l2))


def dist2many(loc, locs):
    # return norm(l1 - l2)
    return np.linalg.norm(locs - loc, axis=1)


def build_PShSv_matrix(vec):
    """Create orientations for channels to be rotated."""
    P = vec / norm(vec)
    SH = np.array([P[1], -P[0], 0])
    SV = np.cross(SH, P)
    return np.array([P, SH, SV])


def rotation_matrix(axis, theta):
    """Return ccw rotation about the given axis by theta radians."""
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def rotate_one(data, orients, uvec):
    """Rotate channel data with respect to uvec."""
    contribs = np.sum(orients * uvec, axis=1)
    return np.sum(data * contribs[np.newaxis, :].T, axis=0)


def apply_rotation(vec, theta, axis, eps=1e-10):
    """Apply rotation_matrix to all vectors."""
    for i in range(3):
        vec[i] = np.dot(rotation_matrix(axis, theta), vec[i])
    vec[np.abs(vec) < eps] = 0


def axes_from_orient(az, dip, roll):
    vec = np.array([[1, 0, 0], [0, 1, 0],  [0, 0, 1]]).astype(float)
    apply_rotation(vec, az, axis=vec[2])
    apply_rotation(vec, dip, axis=vec[1])
    apply_rotation(vec, roll, axis=vec[0])
    return vec


def MedAbsDev(grid):
    return np.median(np.abs(grid - np.median(grid)))


def MinMax(a):
    mn, mx = np.min(a), np.max(a)
    return mn, mx, mx - mn


def read_meta(fname):

    meta = np.loadtxt(fname)
    spacing = meta[6]
    lims = meta[:6].reshape(3, 2)
    shape = (np.diff(lims, axis=1).T[0] // spacing).astype(int)
    # xl, yl, zl = lims
    return lims, spacing, shape


def combine_grids(fles, shape):

    nx, ny, nz = shape

    nfle = len(fles)
    # shape = (nx, nx, nx)
    grids = np.zeros((nfle, nz, ny, nx), dtype=np.float32)

    for i, fn in enumerate(fles):
        # print(fn)
        # xl, yl, zl = lims.reshape(3, 2)
        grid = np.load(fn).reshape(nz, ny, nx)
        grids[i] = grid

    return grids


def combine_grids_nll(fles, shape):

    nx, ny, nz = shape
    nfle = len(fles)
    grids = np.zeros((nfle, nx, ny, nz), dtype=np.float32)

    for i, fn in enumerate(fles):
        grids[i] = np.load(fn).reshape(shape)

    return grids


def xyz_max(grid, lims, spacing):
    # thresh = np.std(grid) * nstd
    iwin = np.argmax(grid)

    pt = np.array(np.unravel_index(iwin, grid.shape))[::-1]
    pt = pt * spacing + lims[:, 0]
    return pt


def xyz_index(loc, lims, spacing):
    # thresh = np.std(grid) * nstd
    shape = (np.diff(lims).T[0] / spacing).astype(int)
    nx, ny, nz = shape
    # x, y, z = loc
    inds = ((loc - lims[:, 0]) / spacing).astype(int)
    ix, iy, iz = inds
    return (iz * nx * ny) + (iy * nx) + ix


def remap(x, out_min, out_max):
    in_min = np.min(x)
    in_max = np.max(x)
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def bandpass(data, band, sr, corners=4, zerophase=True):

    freqmin, freqmax = band
    fe = 0.5 * sr
    low = freqmin / fe
    high = freqmax / fe

    z, p, k = iirfilter(corners, [low, high], btype='band',
                        ftype='butter', output='zpk')
    sos = zpk2sos(z, p, k)
    if zerophase:
        firstpass = sosfilt(sos, data)
        if len(data.shape) == 1:
            return sosfilt(sos, firstpass[::-1])[::-1]
        else:
            return np.fliplr(sosfilt(sos, np.fliplr(firstpass)))
    else:
        return sosfilt(sos, data)


def filter(data, btype, band, sr, corners=4, zerophase=True):
    # btype: lowpass, highpass, band

    fe = 0.5 * sr
    z, p, k = iirfilter(corners, band / fe, btype=btype,
                        ftype='butter', output='zpk')
    sos = zpk2sos(z, p, k)
    if zerophase:
        firstpass = sosfilt(sos, data)
        if len(data.shape) == 1:
            return sosfilt(sos, firstpass[::-1])[::-1]
        else:
            return np.fliplr(sosfilt(sos, np.fliplr(firstpass)))
    else:
        return sosfilt(sos, data)


def decimate(data_in, sr, factor):
    data = data_in.copy()
    fmax = sr / (factor * 2)
    filter(data, 'lowpass', fmax, sr)
    if len(data.shape) == 1:
        return data[::factor]
    else:
        return data[:, ::factor]


def norm2d(d):
    return d / np.max(np.abs(d), axis=1)[:, np.newaxis]


def nextpow2(val):
    buf = math.ceil(math.log(val) / math.log(2))
    return int(math.pow(2, buf))


def cross_corr(sig1, sig2, norm=True, pad=False, phase_only=False, phat=False):
    """Cross-correlate two signals."""
    pad_len = len(sig1)
    if pad is True:
        pad_len *= 2
        # pad_len = signal.next_pow_2(pad_len)

    sig1f = fft(sig1, pad_len)
    sig2f = fft(sig2, pad_len)

    if phase_only is True:
        ccf = np.exp(- 1j * np.angle(sig1f)) * np.exp(1j * np.angle(sig2f))
    else:
        ccf = np.conj(sig1f) * sig2f

    if phat:
        ccf = ccf / np.abs(ccf)

    cc = np.real(ifft(ccf))

    if norm:
        cc /= np.sqrt(energy(sig1) * energy(sig2))

    return np.roll(cc, len(cc) // 2)


def unique_pairs(keys):
    return np.array(list(itertools.combinations(keys, 2)))


def ckeys_remove_intersta(ckeys, names):
    ikeep = []
    for i, ck in enumerate(names[ckeys]):
        n1, n2 = ck
        sta1 = ck[0].split('.')[0]
        sta2 = ck[1].split('.')[0]
        if sta1 == sta2:
            continue
        else:
            ikeep.append(i)

    return ckeys[np.array(ikeep)]


def ckeys_remove_intersta_str(ckeys):
    ikeep = []
    for i, ck in enumerate(ckeys):
        n1, n2 = ck
        sta1 = ck[0].split('.')[0]
        sta2 = ck[1].split('.')[0]
        if sta1 == sta2:
            continue
        else:
            ikeep.append(i)

    return ckeys[np.array(ikeep)]


def pairs_with_autocorr(keys):
    return np.array(list(itertools.combinations_with_replacement(keys, 2)))


def combos_between(keys1, keys2):
    return np.array(list(itertools.product(keys1, keys2)))


def dist_diff_ckeys(ckeys, locs):
    return np.linalg.norm(np.diff(locs[ckeys], axis=1).reshape(-1, 3), axis=1)


def xcorr_ckeys(dat, ckeys, norm=True):

    ncc = len(ckeys)
    nsta, wlen = dat.shape
    padlen = wlen * 2
    # nfreq = int(padlen // 2 + 1)
    # fstack = np.zeros((ncc, nfreq), dtype=np.complex64)
    stack = np.zeros((ncc, padlen), dtype=np.float32)

    fdat = np.fft.rfft(dat, axis=1, n=padlen)
    if norm:
        for irow in range(fdat.shape[0]):
            fdat[irow] /= np.sqrt(energy_freq(fdat[irow]))
            # fdat[irow] = filt * xutil.phase(fdat[irow])

    for j, ckey in enumerate(ckeys):
        k1, k2 = ckey
        stack[j] = np.fft.irfft(np.conj(fdat[k1]) * fdat[k2])

    stack = np.roll(stack, padlen // 2, axis=1)

    return stack


def keeplag(dat, nkeep):
    nsamp = dat.shape[-1]
    hl = int(nsamp // 2)
    return dat[:, hl - nkeep:hl + nkeep]


def energy(sig, axis=None):
    return np.sum(sig ** 2, axis=axis)


def energy_freq(fsig, axis=None):
    return np.sum(np.abs(fsig) ** 2, axis=axis) / fsig.shape[-1]


def norm_energy_freq(fsig):
    val = np.sqrt(np.sum(np.abs(fsig) ** 2) / len(fsig))
    return fsig / val


def build_slice_inds(start, stop, wlen, stepsize=None):

    if stepsize is None:
        stepsize = wlen

    overlap = wlen - stepsize
    imin = np.arange(start, stop - overlap - 1, stepsize)
    imax = np.arange(start + wlen, stop + stepsize - 1, stepsize)
    slices = np.dstack((imin, imax))[0].astype(int)
    if slices[-1][1] > stop:
        slices = slices[:-1]

    return slices


def build_welch_wins(start, stop, wlen, stepsize=None):

    if stepsize is None:
        stepsize = wlen

    overlap = wlen - stepsize
    imin = np.arange(start, stop - overlap - 1, stepsize)
    imax = np.arange(start + wlen, stop + stepsize - 1, stepsize)
    imid = imin + (imax - imin) // 2
    slices = np.dstack((imin, imid, imax))[0].astype(int)
    if slices[-1][1] > stop:
        slices = slices[:-1]

    return slices


def freq_window(cf, npts, sr, norm_energy=True):
    nfreq = int(npts // 2 + 1)
    fsr = npts / sr
    cf = np.array(cf, dtype=float)
    cx = (cf * fsr + 0.5).astype(int)

    win = np.zeros(nfreq, dtype=np.float32)
    win[:cx[0]] = 0
    win[cx[0]:cx[1]] = taper_cosine(cx[1] - cx[0])
    win[cx[1]:cx[2]] = 1
    win[cx[2]:cx[3]] = taper_cosine(cx[3] - cx[2])[::-1]
    win[cx[-1]:] = 0

    if norm_energy is True:
        win /= np.sqrt(energy_freq(win))

    return win


def taper_window(npts, taper_percentage):

    taplen = int(taper_percentage * npts)
    taper = taper_cosine(taplen)

    window = np.ones(npts, dtype=np.float32)
    window[:taplen] *= taper
    window[-taplen:] *= taper[::-1]
    return window


def taper_cosine(wlen):
    return np.cos(np.linspace(np.pi / 2., np.pi, wlen)) ** 2


def phase(sig):
    return np.exp(1j * np.angle(sig))


def whiten2D(a, freqs, sr):
    wl = a.shape[1]
    win = freq_window(freqs, wl, sr)
    af = fft(a)
    for sx in range(a.shape[0]):
        whiten_freq(af[sx], win)
    a[:] = np.real(ifft(af))


def whiten(sig, win):
    """Whiten signal, modified from MSNoise."""
    npts = len(sig)
    nfreq = int(npts // 2 + 1)

    assert(len(win) == nfreq)
    # fsr = npts / sr

    fsig = fft(sig)
    # hl = nfreq // 2

    half = fsig[: nfreq]
    half = win * phase(half)
    fsig[: nfreq] = half
    fsig[-nfreq + 1:] = half[1:].conjugate()[::-1]

    return np.real(ifft(fsig))


def whiten_freq(fsig, win):
    # npts = len(fsig)
    # nfreq = int(npts // 2 + 1)
    nfreq = int(len(fsig) // 2 + 1)
    assert(len(win) == nfreq)
    fsig[: nfreq] = win * phase(fsig[: nfreq])
    fsig[-nfreq + 1:] = fsig[1: nfreq].conjugate()[::-1]


def mirror_freqs(data):
    nfreq = int(data.shape[1] // 2 + 1)
    data[:, -nfreq + 1:] = np.fliplr(data[:, 1: nfreq].conjugate())


def taper_data(data, wlen):
    # tap = taper_cosine(wlen)
    tap = hann_half(wlen)
    data[:wlen] *= tap
    data[-wlen:] *= tap[::-1]


def taper2d(data, wlen):
    out = data.copy()
    tap = hann_half(wlen)
    for i in range(data.shape[0]):
        out[i][:wlen] *= tap
        out[i][-wlen:] *= tap[::-1]
    return out


def amax_cc(sig):
    return np.argmax(sig) - len(sig) // 2


def angle(a, b):
    # return np.arctan((a[1] - b[1]) / (a[0] - b[0]))
    return np.arctan2((a[1] - b[1]), (a[0] - b[0]))


def noise1d(wlen, freqs, sr, scale, taplen=0.05):

    # padlen =
    # fwin = freq_window(freqs, wlen, sr)
    # nfreq = int(npts // 2 + 1)
    # fwin = whiten_window(freqs, nfreq, sr)
    fwin = freq_window(freqs, wlen, sr)
    nfreq = len(fwin)

    fb = np.zeros(nfreq, dtype=np.complex64)
    phases = np.random.rand(nfreq) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)

    fb = phases * fwin
    # out = np.real(np.fft.irfft(fb, n=nfreq)[:wlen]) * scale
    out = np.real(np.fft.irfft(fb, n=wlen)[:wlen]) * scale
    if taplen > 0:
        taper_data(out, int(taplen * wlen))

    return out


# def whiten_window(corners, nfreq, sr, norm_energy=True):

#     # freqs = np.fft.rfftfreq(nfreq, 1.0 / sr)
#     fsr = nfreq / sr
#     cf = np.array(corners, dtype=np.float32)
#     cx = (cf * fsr + 0.5).astype(int)
#     # print(cx)
#     # print(freqs[cx])

#     win = np.zeros(nfreq, dtype=np.float32)
#     win[:cx[0]] = 0
#     win[cx[0]:cx[1]] = taper_cosine(cx[1] - cx[0])
#     win[cx[1]:cx[2]] = 1
#     win[cx[2]:cx[3]] = taper_cosine(cx[3] - cx[2])[::-1]
#     win[cx[-1]:] = 0

#     if norm_energy is True:
#         win /= np.sqrt(energy_freq(win))

#     return win


# def noise1d(npts, freqs, sr, scale, taplen=0.05):

#     out = np.zeros(npts, dtype=np.float32)
#     fwin = freq_window(freqs, npts, sr)
#     nfreq = len(fwin)

#     fb = np.zeros(npts, dtype=np.complex64)
#     fb = np.zeros(npts, dtype=np.complex64)
#     phases = np.random.rand(nfreq) * 2 * np.pi
#     phases = np.cos(phases) + 1j * np.sin(phases)

#     fb[: nfreq] = phases * fwin
#     fb[-nfreq + 1:] = fb[1:nfreq].conjugate()[::-1]
#     # a[i] = np.real(ifft(fb))
#     out += np.real(ifft(fb)) * scale
#     if taplen > 0:
#         taper_data(out, int(taplen * npts))

#     return out


def add_noise(a, freqs, sr, scale, taplen=0.05):
    out = a.copy()

    nsig, npts = a.shape
    fwin = freq_window(freqs, npts, sr)
    nfreq = len(fwin)

    for i in range(nsig):
        fb = np.zeros(npts, dtype=np.complex64)

        phases = np.random.rand(nfreq) * 2 * np.pi
        phases = np.cos(phases) + 1j * np.sin(phases)

        fb[: nfreq] = phases * fwin
        fb[-nfreq + 1:] = fb[1:nfreq].conjugate()[::-1]
        # a[i] = np.real(ifft(fb))
        out[i] += np.real(ifft(fb)) * scale
    taper2d(out, int(taplen * npts))

    return out


def zeropad2d(a, npad):
    nrow, ncol = a.shape
    out = np.zeros((nrow, ncol + npad), dtype=a.dtype)
    out[:, :ncol] = a
    return out


def zero_argmax(a, wlen, taplen=0.05):
    # handles edges incorrectly
    npts = a.shape[1]
    taper = hann_half(int(taplen * wlen))
    win = np.concatenate((taper[::-1], np.zeros(wlen), taper))
    hl = int(len(win) // 2)
    out = a.copy()
    imaxes = np.argmax(np.abs(out), axis=1)
    for i, imax in enumerate(imaxes):
        i0 = imax - hl
        i1 = imax + hl
        if i0 <= 0:
            out[i, 0:i1] *= win[abs(i0):]
        elif i1 > npts:
            out[i, i0:] *= win[:-(i1 - npts)]
        else:
            out[i, i0:i1] *= win

    return out


def hann(npts):
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(npts) / (npts - 1))


def hann_half(npts):
    return hann(npts * 2)[:npts]

# def whiten(sig, win):
#   """Whiten signal, modified from MSNoise."""
#   npts = len(sig)
#   nfreq = int(npts // 2 + 1)

#   assert(len(win) == nfreq)
#   # fsr = npts / sr

#   fsig = fft(sig)
#   # hl = nfreq // 2

#   half = fsig[: nfreq]
#   half = win * phase(half)
#   fsig[: nfreq] = half
#   fsig[-nfreq + 1:] = half[1:].conjugate()[::-1]

#   return np.real(ifft(fsig))


def fftnoise(f):
    f = np.array(f, dtype='complex')
    npts = (len(f) - 1) // 2
    phases = np.random.rand(npts) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1:npts + 1] *= phases
    f[-1:-1 - npts:-1] = np.conj(f[1:npts + 1])
    return ifft(f).real


def band_noise(band, sr, samples):
    freqmin, freqmax = band
    freqs = np.abs(fftfreq(samples, 1. / sr))
    f = np.zeros(samples)
    idx = np.where(np.logical_and(freqs >= freqmin, freqs <= freqmax))[0]
    f[idx] = 1
    return fftnoise(f)


def add_noise2(data, band, sr, power):
    freqmin, freqmax = band
    samples = data.shape[1]
    for d in data:
        d += band_noise(band, sr, samples) * power


# def envelope(data):
#     slen = len(data)
#     FFT = fft(data, slen)
#     FFT[1: slen // 2] *= 2
#     FFT[slen // 2:] = 0
#     return np.abs(ifft(FFT))

def envelope(data, **kwargs):
    return np.abs(hilbert(data, **kwargs))


def smooth(data, nsmooth, **kwargs):
    return gaussian_filter(data, nsmooth, **kwargs)

# def pairs_excluding_acoustic():
        # fd = rfft(sig)
    # # keys = np.arange(ds.shape[0])
    # # ckeys = np.array(list(itertools.combinations(keys, 2))).astype(np.int32)
    # ckeys = ck
    # scut = 700
    # avel = 345.
    # sloc = srcs[0]

    # dd0 = np.array([xutil.dist_between(l, sloc) for l in locs])
    # dd1 = np.diff(dd0[ckeys], axis=1)
    # ia = dd1 / avel * sr
    # # plt.hist(np.abs(ia), bins=100)
    # ik0 = np.where(np.abs(ia) < scut)[0]
    # sloc = srcs[1]
    # dd0 = np.array([xutil.dist_between(l, sloc) for l in locs])
    # dd1 = np.diff(dd0[ckeys], axis=1)
    # ia = dd1 / avel * sr
    # # plt.hist(np.abs(ia), bins=100)
    # ik1 = np.where(np.abs(ia) < scut)[0]
    # ik = np.unique(np.concatenate((ik0, ik1)))
    # ck2 = np.delete(ckeys, ik, axis=0)
    # xplot.stations(locs, ckeys=ck2[::500])
    # dd = np.linalg.norm(np.diff(locs[ck2], axis=1).reshape(-1, 3), axis=1)
    # # plt.hist(dd, bins=100)
    # cki = np.where(dd < 1000)[0]

def build_checkerboard(w, h):
    re = np.r_[w * [0, 1]]              # even-numbered rows
    ro = np.r_[w * [1, 0]]              # odd-numbered rows
    return np.row_stack(h * (re, ro))


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def nans_interp(data, threshold=.05):
    nans, x = nan_helper(data)
    number_of_nans = data[nans].size
    if number_of_nans > 0:
        if float(number_of_nans) / len(data) < threshold:
            data[nans] = np.interp(x(nans), x(~nans), data[~nans])
        else:
            data[:] = 0


# def xcorr_pair_stack_slices(sig1, sig2, cclen, stacklen, keeplag, stepsize=None, whiten_freqs=None, sr=None, onebit=False):

#     cclen = int(cclen)
#     keeplag = int(keeplag)
#     slices = build_slice_inds(0, len(sig1), cclen, stepsize=stepsize)
#     xvals = np.mean(slices, axis=1)

#     ncc = int(slices[-1][1] / stacklen)
#     padlen = cclen * 2
#     nfreq = int(padlen // 2 + 1)
#     # ccs = np.zeros((ncc, padlen), dtype=np.float32)
#     keeplen = int(keeplag * 2)
#     ccs = np.zeros((ncc, keeplen), dtype=np.float32)

#     win = None
#     if whiten_freqs is not None:
#         win = freq_window(whiten_freqs, padlen, sr)

#     step = slices[1][0] - slices[0][0]
#     assert(stacklen % step == 0)
#     chunksize = stacklen // step
#     print(step, chunksize)
#     print(ncc)

#     fcc = np.zeros(nfreq, dtype=np.complex64)
#     tmp_nstack = 0
#     cix = 0

#     for i, sl in enumerate(slices):
#         print(f"{i} / {len(slices)}")
#         win1 = sig1[sl[0]:sl[1]]
#         win2 = sig2[sl[0]:sl[1]]

#         if onebit is True:
#             win1 = np.sign(win1)
#             win2 = np.sign(win2)

#         f1 = np.fft.rfft(win1, n=padlen)
#         f2 = np.fft.rfft(win2, n=padlen)

#         if win is not None:
#             f1 = win * phase(f1)
#             f2 = win * phase(f2)

#         f1 /= np.sqrt(energy_freq(f1))
#         f2 /= np.sqrt(energy_freq(f2))
#         fcc += np.conj(f1) * f2
#         tmp_nstack += 1

#         if tmp_nstack >= chunksize:
#             cc = np.fft.irfft(fcc) / tmp_nstack
#             ccs[cix, :keeplag] = cc[-keeplag:]
#             ccs[cix, keeplag:] = cc[:keeplag]

#             fcc.fill(0)
#             tmp_nstack = 0
#             cix += 1

#     return xvals, ccs


# def xcorr_ckeys_stack_slices(rawdat, ckeys, cclen, keeplag, stepsize=None, whiten_freqs=None, sr=None, onebit=False):

#     cclen = int(cclen)

#     nchan, nsamp = rawdat.shape
#     slices = build_slice_inds(0, nsamp, cclen, stepsize=stepsize)
#     # xvals = np.mean(slices, axis=1)

#     # ncc = int(slices[-1][1] / stacklen)
#     ncc = len(ckeys)
#     padlen = cclen * 2
#     nfreq = int(padlen // 2 + 1)

#     whiten_win = None
#     if whiten_freqs is not None:
#         whiten_win = freq_window(whiten_freqs, padlen, sr)

#     fstack = np.zeros((ncc, nfreq), dtype=np.complex64)

#     for i, sl in enumerate(slices):
#         # print(f"{i} / {len(slices)}")
#         dat = rawdat[:, sl[0]:sl[1]].copy()

#         if onebit is True:
#             dat = bandpass(dat, whiten_freqs[[0, -1]], sr)
#             dat = np.sign(dat)

#         fdat = np.fft.rfft(dat, axis=1, n=padlen)

#         if whiten_win is not None:
#             for irow in range(fdat.shape[0]):
#                 fdat[irow] = whiten_win * phase(fdat[irow])
#         else:
#             for irow in range(fdat.shape[0]):
#                 fdat[irow] /= np.sqrt(energy_freq(fdat[irow]))

#         # dat0 = np.fft.irfft(fdat, axis=1)
#         # norm = xutil.energy_freq(fdat[0])

#         for j, ckey in enumerate(ckeys):
#             k1, k2 = ckey
#             # stack[j] = np.fft.irfft(np.conj(fdat[k1]) * fdat[k2])
#             fstack[j] += np.conj(fdat[k1]) * fdat[k2]

#     # stack = np.fft.irfft(fstack, axis=1)
#     # stack = np.roll(stack, padlen // 2, axis=1)
#     keeplag = int(keeplag)
#     stack = np.zeros((ncc, keeplag * 2), dtype=np.float32)

#     for i in range(ncc):
#         cc = np.fft.irfft(fstack[i]) / len(slices)
#         stack[i, :keeplag] = cc[-keeplag:]
#         stack[i, keeplag:] = cc[:keeplag]

#     return stack


# def xcorr_freq(sig1f, sig2f):
#   """Cross-correlate two signals."""

#   ccf = np.conj(sig1f) * sig2f

#   if phat:
#       ccf = ccf / np.abs(ccf)

#   cc = np.real(ifft(ccf))

#   if norm:
#       cc /= np.sqrt(energy(sig1) * energy(sig2))

#   return np.roll(cc, len(cc) // 2)
