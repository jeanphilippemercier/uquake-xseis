#pragma once

// #include <iostream>
// #include <vector>
// #include <iterator>
// #include <numeric>
// #include <string>
#include "xseis2/core.h"
#include "xseis2/logger.h"
#include "xseis2/hdf5.h"
#include "xseis2/npy.h"
#include "xseis2/signal.h"
#include "xseis2/keygen.h"
#include "xseis2/beamform.h"

#include <omp.h>
#include <fftw3.h>


namespace xseis {


void WFSearchOneVel(Array2D<float>& rdat, float sr, Array2D<float>& stalocs, Vector<uint16_t>& chanmap, Array2D<uint16_t>& ttable, int64_t* outbuf, std::string& file_out, int debug) 
{	
	uint32_t const smooth_cc_wlen = 10;
	auto logger = xseis::Logger();
	// PREPROC ///////////////////////////////////////////////////////////////////////
	auto dat = xseis::ZeroPad(rdat, rdat.ncol() * 2);
	logger.log("create padded");

	auto fdat = xseis::WhitenAndFFT(dat, sr, {40, 50, 350, 360});
	logger.log("fft");
	xseis::NpzSave(file_out, "sigs_preproc", dat.rows(), "w");
	logger.log("save dat");

	auto groups = xseis::GroupChannels(chanmap.span());

	auto keys = xseis::Arange<uint16_t>(0, stalocs.nrow());
	auto pairs_all = xseis::UniquePairs(keys);
	// auto pairs = xseis::DistFiltPairs(pairs_all.rows(), stalocs.rows(), 300, 1500);
	auto pairs = pairs_all.rows();

	xseis::NpzSave(file_out, "sta_ckeys", pairs_all.rows(), "a");
	std::cout << "pairs_all.nrow(): " << pairs_all.nrow() << '\n';
	std::cout << "npairs: " << pairs.size() << '\n';
	logger.log("keygen");

	auto ccdat = xseis::Array2D<float>(pairs.size(), dat.ncol());	
	xseis::XCorrChanGroupsAbs(fdat, groups, pairs, ccdat, smooth_cc_wlen);
	logger.log("xcorr");
	xseis::NpzSave(file_out, "sigs_xcorrs", ccdat.rows(), "a");
	logger.log("save ccfs");	
	auto rmaxes = xseis::RowMaxes(ccdat.rows());
	auto vmax_cc = xseis::Mean(gsl::make_span(rmaxes));
	logger.log("vmax_cc");

	auto power = xseis::Vector<float>(ttable.ncol());
	xseis::InterLocBlocks(ccdat.rows(), pairs, ttable.rows(), power.span());
	logger.log("interloc");
	xseis::NpzSave(file_out, "grid_power", power.span(), "a");

	auto imax = xseis::ArgMax(power.span());
	auto vmax = xseis::Max(power.span());
	// std::cout << "wloc: " << points.span(imax) << "\n";
	std::cout << "(vmax/theor): " << vmax << " / " << vmax_cc << " = " << vmax / vmax_cc * 100.0f << "% \n";

	auto wtt = ttable.copy_col(imax);
	xseis::NpzSave(file_out, "tts_to_max", wtt.span(), "a");
	logger.log("wtt");

	// xseis::NpySave(dir_dat + "d1.npy", dat);
	// xseis::RollAndStack(dat, groups, wtt.span());
	xseis::Envelope(dat.rows());
	logger.log("envelope");	
	xseis::RollSigs(dat.rows(), groups, wtt.span());
	logger.log("roll");
	auto stack = xseis::StackSigs(dat.rows());
	logger.log("stack");
	
	xseis::NpzSave(file_out, "sigs_rolled", dat.rows(), "a");
	xseis::NpzSave(file_out, "sig_stack", stack.span(), "a");

	auto otime = xseis::ArgMax(stack.span());
	std::cout << "otime: " << otime << "\n";
	std::cout << "imax: " << imax << "\n";

	outbuf[0] = vmax * 10000;
	outbuf[1] = imax;
	outbuf[2] = otime; // ot ix for original sr

	logger.summary();

}


void SearchOnePhase(float* rawdat_p, uint32_t nchan, uint32_t npts, float sr, float* stalocs_p, uint32_t nsta, uint16_t* chanmap_p, uint16_t* ttable_ptr, uint32_t ngrid, int64_t* outbuf, uint32_t nthreads, std::string& file_out, int debug) 
{

	omp_set_num_threads(nthreads);
	std::string file_wisdom = "fftw3_wisdom.txt";
	fftwf_import_wisdom_from_filename(&file_wisdom[0]);
	
	auto dat = Array2D<float>(rawdat_p, nchan, npts, true); // copies data
	auto stalocs = Array2D<float>(stalocs_p, nsta, 3, false);
	auto chanmap = Vector<uint16_t>(chanmap_p, nchan);
	auto ttable = Array2D<uint16_t>(ttable_ptr, nsta, ngrid, false);
	// auto ttable2 = Array2D<uint16_t>(ttable2_ptr, nsta, ngrid);
	// auto grid = Vector<float>(grid_p, ngrid);
	// CorrSearchDec2XBoth(rawdat, sr, stalocs, chanmap, ttable1, ttable2, outbuf, nthreads, logdir, debug);

	WFSearchOneVel(dat, sr, stalocs, chanmap, ttable, outbuf, file_out, debug);

	// InterLocDec2X(Array2D<float> &dat, float sr, Array2D<float> &stalocs, Vector<uint16_t> &chanmap, Array2D<uint16_t> &ttable, uint32_t *outbuf, Vector<float> &grid, std::string &file_out, int debug)

	fftwf_export_wisdom_to_filename(&file_wisdom[0]);

	
}




void SearchWinDec2X(Array2D<float>& dat, float sr, Array2D<float>& stalocs, Vector<uint16_t>& chanmap, Array2D<uint16_t>& ttable, uint32_t* outbuf, std::string& file_out, int debug) 
{	
	auto logger = xseis::Logger();
	// PREPROC ///////////////////////////////////////////////////////////////////////
	auto fdat = xseis::WhitenAndFFTPadDec2x(dat, sr, {40, 50, 350, 360});
	logger.log("fft");
	xseis::NpzSave(file_out, "sigs_preproc", dat.rows(), "w");
	logger.log("save dat");

	// Correlate ///////////////////////////////////////////////////////////////////
	// group similar channels and create cc pairs
	auto keys = xseis::Arange<uint16_t>(0, stalocs.nrow());
	// auto groups = xseis::GroupChannels(keys, chanmap);
	auto groups = xseis::GroupChannels(chanmap.span());

	auto pairs_all = xseis::UniquePairs(keys);
	// auto pairs = xseis::DistFiltPairs(pairs_all.rows(), stalocs.rows(), 300, 1500);
	auto pairs = pairs_all.rows();

	xseis::NpzSave(file_out, "sta_ckeys", pairs_all.rows(), "a");
	std::cout << "pairs_all.nrow(): " << pairs_all.nrow() << '\n';
	std::cout << "npairs: " << pairs.size() << '\n';
	logger.log("keygen");
	// return 0;

	auto ccdat = xseis::Array2D<float>(pairs.size(), dat.ncol());	
	// xseis::XCorrChanGroupsEnvelope(fdat, groups, pairs, ccdat);
	xseis::XCorrChanGroupsAbs(fdat, groups, pairs, ccdat);
	logger.log("xcorr");
	xseis::NpzSave(file_out, "sigs_xcorrs", ccdat.rows(), "a");
	logger.log("save ccfs");	
	auto rmaxes = xseis::RowMaxes(ccdat.rows());
	auto vmax_cc = xseis::Mean(gsl::make_span(rmaxes));
	logger.log("vmax_cc");

	auto power = xseis::Vector<float>(ttable.ncol());
	xseis::InterLocBlocks(ccdat.rows(), pairs, ttable.rows(), power.span());
	logger.log("interloc");
	xseis::NpzSave(file_out, "grid_power", power.span(), "a");

	auto imax = xseis::ArgMax(power.span());
	auto vmax = xseis::Max(power.span());
	// std::cout << "wloc: " << points.span(imax) << "\n";
	std::cout << "(vmax/theor): " << vmax << " / " << vmax_cc << " = " << vmax / vmax_cc * 100.0f << "% \n";

	auto wtt = ttable.copy_col(imax);
	xseis::NpzSave(file_out, "tts_to_max", wtt.span(), "a");
	logger.log("wtt");

	// xseis::NpySave(dir_dat + "d1.npy", dat);
	// xseis::RollAndStack(dat, groups, wtt.span());
	xseis::Envelope(dat.rows());
	logger.log("envelope");	
	xseis::RollSigs(dat.rows(), groups, wtt.span());
	logger.log("roll");
	auto stack = xseis::StackSigs(dat.rows());
	logger.log("stack");
	
	xseis::NpzSave(file_out, "sigs_rolled", dat.rows(), "a");
	xseis::NpzSave(file_out, "sig_stack", stack.span(), "a");

	auto otime = xseis::ArgMax(stack.span()) * 2;
	std::cout << "otime: " << otime << "\n";

	outbuf[0] = vmax * 10000;
	outbuf[1] = imax;
	outbuf[2] = otime; // ot ix for original sr

	logger.summary();

}


void SearchWinDec2X(float* rawdat_p, uint32_t nchan, uint32_t npts, float sr, float* stalocs_p, uint32_t nsta, uint16_t* chanmap_p, uint16_t* ttable_ptr, uint32_t ngrid, uint32_t* outbuf, uint32_t nthreads, std::string& file_out, int debug) 
{

	omp_set_num_threads(nthreads);
	std::string file_wisdom = "fftw3_wisdom.txt";
	fftwf_import_wisdom_from_filename(&file_wisdom[0]);
	
	auto dat = Array2D<float>(rawdat_p, nchan, npts, true); // copies data
	auto stalocs = Array2D<float>(stalocs_p, nsta, 3, false);
	auto chanmap = Vector<uint16_t>(chanmap_p, nchan);
	auto ttable = Array2D<uint16_t>(ttable_ptr, nsta, ngrid, false);
	// auto ttable2 = Array2D<uint16_t>(ttable2_ptr, nsta, ngrid);
	// auto grid = Vector<float>(grid_p, ngrid);
	// CorrSearchDec2XBoth(rawdat, sr, stalocs, chanmap, ttable1, ttable2, outbuf, nthreads, logdir, debug);

	SearchWinDec2X(dat, sr, stalocs, chanmap, ttable, outbuf, file_out, debug);

	// InterLocDec2X(Array2D<float> &dat, float sr, Array2D<float> &stalocs, Vector<uint16_t> &chanmap, Array2D<uint16_t> &ttable, uint32_t *outbuf, Vector<float> &grid, std::string &file_out, int debug)

	fftwf_export_wisdom_to_filename(&file_wisdom[0]);

	
}





}

