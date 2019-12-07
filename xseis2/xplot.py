"""
Plotting
"""

# from matplotlib.colors import LogNorm
# from matplotlib.ticker import ScalarFormatter
import matplotlib.pyplot as plt
# from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import datetime
from scipy import fftpack
from scipy.fftpack import fft, ifft, rfft, fftfreq
from xseis2 import xutil
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path

from matplotlib.pyplot import rcParams
rcParams['figure.figsize'] = 11, 8


def axvline_times(times, mirror=True, **kwargs):
    for x in times:
        axvline(x, **kwargs)
        if mirror:
            axvline(-x, **kwargs)


def axvline(*args, **kwargs):
    defs = {'linestyle': '--', 'alpha': 0.5, "color": 'red'}
    for k, v in defs.items():
        if k not in kwargs:
            kwargs[k] = v

    plt.axvline(*args, **kwargs)


def axhline(*args, **kwargs):
    defs = {'linestyle': '--', 'alpha': 0.5, "color": 'red'}
    for k, v in defs.items():
        if k not in kwargs:
            kwargs[k] = v

    plt.axhline(*args, **kwargs)


def quicksave(fig=None, savedir=None, prefix='py', dpi=100):
    if fig is None:
        fig = plt.gcf()
    if savedir is None:
        savedir = os.path.join(os.environ['HOME'], "Pictures", "ot")
    tstamp = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
    fname = f"{prefix}_{tstamp}.png"
    fpath = os.path.join(savedir, fname)
    plt.tight_layout()

    fig.savefig(fpath, dpi=dpi)


def moveout(dat, dists, sr=1, scale=20, picks=None):

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    dshift = xutil.norm2d(dat) * 20 + dists[:, np.newaxis]

    dlines = ax.plot(dshift.T, alpha=0.3, color='black')
    dline = dlines[0]
    plt.title("sr: %.2f" % sr)
    plt.xlabel('time (sample)')
    plt.ylabel('dist (m)')

    if picks is not None:
        for i, pick in enumerate(picks):
            plt.scatter(pick, dists, s=50,
                 marker='|', zorder=1, label="picks_%d" % i)
    # plt.legend()
    lpick = []

    def handler(event):
        lpick.append([event.xdata, event.ydata])
        x, y = lpick[-1]
        # print("%d, %d" % (x, y))
        ax.scatter(x, y, s=30)
        if len(lpick) == 2:
            xv, yv = np.array(lpick).T
            xd = xv[1] - xv[0]
            yd = yv[1] - yv[0]
            vel = yd / xd * sr
            ax.plot(xv, yv)
            ax.text(x, y, "%d" % vel, size=20, color='red')
            lpick.clear()
        dline.figure.canvas.draw()

    cid = dline.figure.canvas.mpl_connect('key_press_event', handler)


def chan_groups(d, groups, shifts=None, labels=None, **kwargs):
    nsta = len(groups)
    if shifts is None:
        shifts = np.arange(0, nsta, 1) * 1.0
    ax = plt.gca()
    for i, group in enumerate(groups):
        sig = np.mean(np.abs(d[group]), axis=0)
        tmp = sig / np.max(np.abs(sig)) + shifts[i]
        plt.plot(tmp, **kwargs)
        # ax.set_prop_cycle(None)
        # for key in group:
        #   sig = d[key]
        #   tmp = sig / np.max(np.abs(sig)) + shifts[i]
        #   plt.plot(tmp, **kwargs)

    if labels is not None:
        for i, lbl in enumerate(labels):
            plt.text(0, shifts[i] + 0.1, lbl, fontsize=15)


def sigs(d, shifts=None, labels=None, picks=None, spacing=1.2, **kwargs):

    import matplotlib.lines as mlines

    if shifts is None:
        shifts = np.arange(0, d.shape[0], 1) * spacing
    for i, sig in enumerate(d):
        tmp = sig / np.max(np.abs(sig)) + shifts[i]
        plt.plot(tmp, **kwargs)

    if labels is not None:
        for i, lbl in enumerate(labels):
            plt.text(0, shifts[i] + 0.1, lbl, fontsize=10)

    if picks is not None:
        leg = []
        size = (shifts[1] - shifts[0]) / 4
        for i, pgroup in enumerate(picks):
            clr = "C{}".format(i)
            for j, pick in enumerate(pgroup):
                xv = [pick, pick]
                yv = [shifts[j] - size, shifts[j] + size]
                plt.plot(xv, yv, color=clr)
            leg.append(mlines.Line2D([], [], color=clr, label="picks_%d" % i))

        plt.legend(handles=leg)


def ccf(d, sr=None):
    N = len(d)
    x = np.linspace(-N / 2, N / 2, N)
    if sr is not None:
        plt.plot(x / sr * 1000., d)
        plt.xlabel("Lag time (ms)")
    else:
        plt.plot(x, d)
        plt.xlabel("Lag time (sample)")
    plt.ylabel("Corr")


def v2color(vals):

    cnorm = plt.Normalize(vmin=np.nanmin(vals), vmax=np.nanmax(vals))
    cmap = plt.cm.ScalarMappable(norm=cnorm, cmap=plt.get_cmap('viridis'))
    clrs = [cmap.to_rgba(v) for v in vals]
    return clrs


def build_cmap(vals):

    cnorm = plt.Normalize(vmin=np.nanmin(vals), vmax=np.nanmax(vals))
    cmap = plt.cm.ScalarMappable(norm=cnorm, cmap=plt.get_cmap('viridis'))
    clrs = [cmap.to_rgba(v) for v in vals]
    return clrs, cmap


def stations(locs, lvals=None, ckeys=None, cvals=None, alpha=0.3, lstep=100, pkeys=None, plocs=None):
    locs = locs[:, :2]
    x, y = locs.T
    if lvals is not None:
        plt.scatter(x, y, alpha=alpha, c=lvals, s=100, zorder=0)
    else:
        plt.scatter(x, y, alpha=alpha, s=6, zorder=0)
    # x, y, z = locs[2900:3100].T
    if lstep != 0:
        for i in range(0, locs.shape[0], lstep):
            plt.text(x[i], y[i], i)

    if ckeys is not None:
        if cvals is not None:
            clrs = v2color(cvals)
            for i, ck in enumerate(ckeys):
                x, y = locs[ck].T
                plt.plot(x, y, alpha=alpha, color=clrs[i], linewidth=2)
        else:
            for ck in ckeys:
                x, y = locs[ck].T
                plt.plot(x, y, alpha=alpha, color='black', zorder=1)

    if pkeys is not None:
        x, y = locs[pkeys].T
        plt.scatter(x, y, s=60, color='red', zorder=2)
        for i in range(x.size):
            plt.text(x[i], y[i], i, color='green')

    if plocs is not None:
        x, y = plocs[:, :2].T
        plt.scatter(x, y, s=60, color='red', marker='x', zorder=2)
        for i in range(x.size):
            plt.text(x[i], y[i], i, color='green')

    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.axis('equal')
    plt.show()


def im_freq(d, sr, norm=False, xlims=None):

    fd = fftpack.rfft(d, axis=1)
    fd = np.abs(fd)

    if norm is True:
        fd /= np.max(fd, axis=1)[:, np.newaxis]

    n = fd.shape[1]
    freq = fftpack.rfftfreq(n, d=1. / sr)

    im = plt.imshow(fd, aspect='auto', extent=[freq[0], freq[-1], 0, fd.shape[0]], origin='lower', interpolation='none')
    if xlims is not None:
        plt.xlim(xlims)

    plt.xlabel('Freq (Hz)')
    plt.ylabel('Signal #')
    plt.tight_layout()

    return im


def im(d, norm=True, savedir=None, tkey='im_raw', cmap='viridis', aspect='auto', extent=None, locs=None, labels=None, times=None, title=None):

    fig = plt.figure(figsize=(12, 8))
    if times is not None:
        extent = [times[0], times[-1], 0, d.shape[0]]

    if norm is True:
        dtmp = d / np.max(np.abs(d), axis=1)[:, np.newaxis]
    else:
        dtmp = d
    im = plt.imshow(dtmp, origin='lower', aspect=aspect, extent=extent, cmap=cmap, interpolation='none')
    if extent is not None:
        plt.xlim(extent[:2])
        plt.ylim(extent[2:])
    if locs is not None:
        plt.scatter(locs[:, 0], locs[:, 1])
    plt.colorbar(im)
    if labels is not None:
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
    if title is not None:
        plt.title(title)
    # manager = plt.get_current_fig_manager()
    # manager.resize(*manager.window.maxsize())
    plt.tight_layout()

    return im


def im_ax(d, ax, norm=True, cmap='viridis', aspect='auto', extent=None):

    if norm is True:
        dtmp = d / np.max(np.abs(d), axis=1)[:, np.newaxis]
    else:
        dtmp = d
    im = ax.imshow(dtmp, origin='lower', aspect=aspect, extent=extent,
               cmap=cmap, interpolation='none')
    if extent is not None:
        ax.set_xlim(extent[:2])
        ax.set_ylim(extent[2:])


def freq_compare(sigs, sr, xlim=None):

    plt.subplot(211)
    for sig in sigs:
        plt.plot(sig)
    plt.xlabel('Time')
    plt.subplot(212)
    for sig in sigs:
        f = fftpack.fft(sig)
        freq = fftpack.fftfreq(len(f), d=1. / sr)
        # freq = np.fft.fftshift(freq)
        plt.plot(np.fft.fftshift(freq), np.fft.fftshift(np.abs(f)))
    if xlim is not None:
        plt.xlim(xlim)
    else:
        plt.xlim([0, sr / 2.])
    plt.xlabel('Freq (Hz)')
    plt.show()


def freq(sig, sr, xlim=None, zpad=True):

    plt.subplot(211)
    times = np.linspace(0, len(sig) / sr, len(sig)) * 1000.
    plt.plot(times, sig, marker='o', alpha=1, markersize=0)
    plt.xlabel('Time (ms)')

    plt.subplot(212)
    padlen = len(sig)
    if zpad:
        padlen *= 2

    fsig = np.fft.rfft(sig, n=padlen)
    freqs = np.fft.rfftfreq(padlen, 1.0 / sr)

    plt.plot(freqs, np.abs(fsig), marker='o', alpha=1, markersize=0)
    plt.ylabel('abs(f)')

    # if xlim is not None:
    #     plt.xlim(xlim)
    # else:
    #     plt.xlim([0, sr / 2.])
    plt.xlabel('Freq (Hz)')


def angle(sig, sr, xlim=None):

    plt.subplot(211)
    plt.plot(sig)
    plt.xlabel('Time')
    plt.subplot(212)

    size = len(sig)
    hl = size // 2
    freq = fftpack.fftfreq(size, d=1. / sr)[:hl]
    f = fftpack.fft(sig)[:hl]
    plt.plot(freq, np.abs(f))

    ang = np.angle(f)
    plt.plot(freq, ang)
    if xlim is not None:
        plt.xlim(xlim)
    else:
        plt.xlim([0, sr / 2.])

    plt.xlabel('Freq (Hz)')
    plt.show()


def sigs_old(d, spacing=10, labels=None, vlines=None):

    if vlines is not None:
        for v in vlines:
            plt.axvline(v, linestyle='--', color='red')

    std = np.std(d)
    shifts = np.arange(0, d.shape[0], 1) * spacing * std
    for i, sig in enumerate(d):
        plt.plot(sig + shifts[i])

    if labels is not None:
        for i, lbl in enumerate(labels):
            plt.text(0, shifts[i] + 2 * std, lbl, fontsize=15)

    plt.show()


def sigsNorm(d, spacing=1, labels=None, vlines=None):

    if vlines is not None:
        for v in vlines:
            plt.axvline(v, linestyle='--', color='red')

    shifts = np.arange(0, d.shape[0], 1) * spacing
    for i, sig in enumerate(d):
        plt.plot(sig / np.max(np.abs(sig)) + shifts[i])

    if labels is not None:
        for i, lbl in enumerate(labels):
            plt.text(0, shifts[i], lbl, fontsize=15)

    plt.show()


def savefig(fig, savedir, tkey, dpi=100, facecolor='white', transparent=False):

    if savedir is not None:
        fname = tkey + '.png'
        fpath = os.path.join(savedir, fname)
        # plt.savefig(fpath, dpi=dpi, facecolor=facecolor, transparent=transparent, edgecolor='none')
        fig.savefig(fpath, dpi=dpi)
        plt.close()
    else:
        # fig.show()
        plt.show()

    # plt.close('all')


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot3d(locs, locs2=None):

    fig = plt.figure(figsize=(8, 7), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = locs.T
    ax.scatter(x, y, z, c='green')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')

    if locs2 is not None:
        x, y, z = locs2.T
        ax.scatter(x, y, z, c='red')


def spectro(sig, wl, sr, stepsize=None, norm=False):

    if stepsize is None:
        stepsize = wl // 2

    npts = len(sig)
    slices = xutil.build_slice_inds(0, npts, wl, stepsize=stepsize)
    nsl = len(slices)
    df = np.zeros((nsl, wl), dtype=np.complex)

    for i, sl in enumerate(slices):
        df[i] = fft(sig[sl[0]:sl[1]])

    plt.subplot(212)
    plt.plot(sig)
    plt.xlabel('time')
    plt.xlim([0, npts])
    plt.subplot(211)
    freqs = fftfreq(wl, d=1. / sr)
    # plt.imshow(np.abs(df), aspect='auto', extent=extent, origin='lower', interpolation='none')
    fsr = wl / sr
    imd = np.abs(df[:, : wl // 3]).T
    extent = [0, df.shape[0], freqs[0], (wl // 3) / fsr]

    if norm:
        imd /= np.max(imd, axis=0)
    plt.imshow(imd, aspect='auto', origin='lower', interpolation='none', extent=extent)
    plt.ylabel('freq (hz)')


def ax_spectro(ax, sig, wl, sr, stepsize=None, norm=False):

    if stepsize is None:
        stepsize = wl // 2

    npts = len(sig)
    slices = xutil.build_slice_inds(0, npts, wl, stepsize=stepsize)
    nsl = len(slices)
    df = np.zeros((nsl, wl), dtype=np.complex)

    for i, sl in enumerate(slices):
        df[i] = fft(sig[sl[0]:sl[1]])

    freqs = fftfreq(wl, d=1. / sr)
    # plt.imshow(np.abs(df), aspect='auto', extent=extent, origin='lower', interpolation='none')
    fsr = wl / sr
    imd = np.abs(df[:, : wl // 3]).T
    extent = [0, df.shape[0], freqs[0], (wl // 3) / fsr]

    if norm:
        imd /= np.max(imd, axis=0)
    ax.imshow(imd, aspect='auto', origin='lower', interpolation='none', extent=extent)


class HookImageMax:
    def __init__(self, iplot, factor=0.5):

        self.factor = factor
        self.dline = iplot
        self.ax = iplot.axes
        self.cid = self.dline.figure.canvas.mpl_connect('key_press_event', self)

    def __call__(self, event):

        # ax = self.ax
        factor = self.factor

        if event.key == 'down':
            vmin, vmax = self.dline.get_clim()
            self.dline.set_clim(vmin=factor * vmin, vmax=factor * vmax)
        if event.key == 'up':
            fup = 1 / factor
            vmin, vmax = self.dline.get_clim()
            self.dline.set_clim(vmin=fup * vmin, vmax=fup * vmax)

        self.dline.figure.canvas.draw()


class HookLasso:
    def __init__(self, collection):

        self.collection = collection
        self.dline = collection

        self.ax = collection.axes
        self.alpha_other = 0.3

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        self.cid = self.dline.figure.canvas.mpl_connect('key_press_event', self)
        self.canvas = self.dline.figure.canvas

        self.lasso = LassoSelector(self.ax, onselect=self.onselect)
        self.ind = []

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

    def __call__(self, event):

        if event.key == 'enter':
            print("Selected inds:")
            # print(self.xys[self.ind])
            print(self.ind)
            self.disconnect()

        self.dline.figure.canvas.draw()

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()
