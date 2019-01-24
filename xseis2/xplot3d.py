
import numpy as np
# import math
# from scipy.fftpack import fft, ifft, fftfreq
# import os
# import pickle
import matplotlib.pyplot as plt
from mayavi import mlab
# import matplotlib.gridspec as gridspec
# import subprocess
# import h5py
# import glob
from xseis2 import xutil


def mlab_contour(g, ncont=10, vfmin=0.1, vfmax=0.1, ranges=None, vlims=None, cbar=True):

	if vlims is None:
		gmin, gmax, p2p = xutil.MinMax(g)
		vmin, vmax = gmin + p2p * vfmin, gmax - vfmax * p2p
	else:
		vmin, vmax = vlims
	contours = list(np.linspace(vmin, vmax, ncont))
	src = mlab.pipeline.scalar_field(g)
	# mlab.pipeline.iso_surface(src, contours=contours, opacity=0.3, colormap='viridis', vmin=vmin, vmax=vmax)
	mlab.outline()
	# mlab.axes(line_width=0.5, xlabel='Z', ylabel='Y', zlabel='X', ranges=ranges)
	mlab.axes(line_width=0.5, xlabel='X', ylabel='Y', zlabel='Z', ranges=ranges)
	mlab.pipeline.iso_surface(src, contours=contours[:-1], opacity=0.2, colormap='viridis', vmin=vmin, vmax=vmax)
	mlab.pipeline.iso_surface(src, contours=contours[-1:], opacity=0.8, colormap='viridis', vmin=vmin, vmax=vmax)
	if cbar:
		mlab.colorbar(orientation='vertical')
	return src


# def power(output, gdef, stalocs=None, labels=None, lines=None, lmax=None, title=None):

def power(output, shape, origin, spacing, stalocs=None, labels=None, lines=None, lmax=None, title=None):
	# shape, origin, spacing = gdef[:3], gdef[3:6], gdef[6]
	grid = output.reshape(shape)

	lims = np.zeros((3, 2))
	lims[:, 0] = origin
	lims[:, 1] = origin + shape * spacing
	lims[0] -= lims[0, 0]
	lims[1] -= lims[1, 0]

	fig = mlab.figure(size=(1000, 901))
	ranges = list(lims.flatten())
	src = mlab_contour(grid, ncont=8, vfmin=0.3, vfmax=0.02, ranges=ranges)
	# x, y, z = (sloc - origin) / spacing
	if stalocs is not None:
		x, y, z = (stalocs - origin).T / spacing
		mlab.points3d(x, y, z, color=(1, 0, 0), scale_factor=1.0)
		if labels is not None:
			for i, lbl in enumerate(labels):
				x, y, z = (stalocs[i] - origin) / spacing
				mlab.text3d(x, y, z, str(lbl))

	if lmax is not None:
		x, y, z = (lmax - origin) / spacing
		mlab.points3d(x, y, z, color=(0, 0, 0), scale_factor=2.0)

	lclr = (0.1, 0.1, 0.1)
	if lines is not None:
		for i, line in enumerate(lines):
			x, y, z = (line - origin).T / spacing
			mlab.plot3d(x, y, z, tube_radius=0.2, color=lclr)
	if title is not None:
		mlab.title(title, size=0.5)


def events(locs, vals, stalocs=None, labels=None, lines=None, lmax=None, title=None):

	fig = mlab.figure(size=(1000, 901))
	x, y, z = locs.T
	mlab.points3d(x, y, z, vals, colormap='viridis', scale_factor=4.0)
	mlab.colorbar(orientation='vertical')

	if stalocs is not None:
		x, y, z = stalocs.T
		mlab.points3d(x, y, z, color=(1, 0, 0), scale_factor=20.0, mode='cube')
		mlab.outline()
		mlab.axes(line_width=0.5, xlabel='X', ylabel='Y', zlabel='Z')
		if labels is not None:
			for i, lbl in enumerate(labels):
				x, y, z = stalocs[i]
				mlab.text3d(x, y, z, str(lbl), scale=35.0)


	lclr = (0.1, 0.1, 0.1)
	if lines is not None:
		for i, line in enumerate(lines):
			x, y, z = line
			mlab.plot3d(x, y, z, tube_radius=0.2, color=lclr)
	if title is not None:
		mlab.title(title, size=0.5)


def power_lims(output, gdef, stalocs=None, lmax=None, lines=None, labels=None):

	lims, spacing = gdef[:6], gdef[6]
	shape = (np.diff(lims)[::2] // spacing).astype(int)
	grid = output.reshape(shape)
	origin = lims[::2]

	fig = mlab.figure(size=(1000, 901))
	# fig = mlab.figure()
	ranges = list(lims.flatten())
	# src = mlab_contour(grid, ncont=6, vfmin=0.1, vfmax=0.0, ranges=ranges)
	src = mlab_contour(grid, ncont=6, vfmin=0.1, vfmax=0.01, ranges=ranges)
	# x, y, z = (sloc - origin) / spacing
	if stalocs is not None:
		x, y, z = (stalocs - origin).T / spacing
		mlab.points3d(x, y, z, color=(1, 0, 0), scale_factor=1.0)
		if labels is not None:
			for i, lbl in enumerate(labels):
				x, y, z = (stalocs[i] - origin) / spacing
				mlab.text3d(x, y, z, str(lbl))
				# mlab.plot3d(x, y, z, tube_radius=0.2, color=lclr)

	if lmax is not None:
		x, y, z = (lmax - origin) / spacing
		mlab.points3d(x, y, z, color=(0, 0, 0), scale_factor=2.0)

	# mlab.view(azimuth=-130, elevation=111, distance='auto', focalpoint='auto',   roll=29)
	lclr = (0.1, 0.1, 0.1)
	if lines is not None:
		for i, line in enumerate(lines):
			x, y, z = (line - origin).T / spacing
			mlab.plot3d(x, y, z, tube_radius=0.2, color=lclr)


def power_new(output, gdef, stalocs=None, labels=None, lines=None, lmax=None, title=None, vfmin=0.1, vfmax=0.005):

	lims, spacing = gdef[:6], gdef[6]
	shape = (np.diff(lims)[::2] // spacing).astype(int)
	grid = output.reshape(shape)
	origin = lims[::2]

	fig = mlab.figure(size=(1000, 901))
	ranges = list(lims.flatten())
	src = mlab_contour(grid, ncont=8, vfmin=vfmin, vfmax=vfmax, ranges=ranges)
	# x, y, z = (sloc - origin) / spacing
	if stalocs is not None:
		x, y, z = (stalocs - origin).T / spacing
		mlab.points3d(x, y, z, color=(1, 0, 0), scale_factor=1.0)
		if labels is not None:
			for i, lbl in enumerate(labels):
				x, y, z = (stalocs[i] - origin) / spacing
				mlab.text3d(x, y, z, str(lbl))

	if lmax is not None:
		x, y, z = (lmax - origin) / spacing
		mlab.points3d(x, y, z, color=(0, 0, 0), scale_factor=2.0)

	lclr = (0.1, 0.1, 0.1)
	if lines is not None:
		for i, line in enumerate(lines):
			x, y, z = (line - origin).T / spacing
			mlab.plot3d(x, y, z, tube_radius=0.2, color=lclr)
	if title is not None:
		mlab.title(title, size=0.5)
