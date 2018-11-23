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


namespace xs {

void WFSearchOnePhase(Array2D<float>& rdat, float sr, Array2D<float>& stalocs, Vector<uint16_t>& chanmap, VecOfSpans<uint16_t> ttable, int64_t* outbuf, int debug, std::string& npzfile) 
{	

	std::vector<float> cfreqs {40, 50, 350, 360};
	float dist_min = 0;
	float dist_max = 2000;
	uint32_t const smooth_cc_wlen = 0.005 * sr;

	auto logger = xs::Logger();
	if (debug > 0) logger.log("Start");

	// PREPROC //////////////////////////////////////////////////////////////
	auto dat = xs::ZeroPad(rdat, rdat.ncol() * 2);
	if (debug > 0) logger.log("create padded");

	auto fdat = xs::FFTAndWhiten(dat, sr, cfreqs);
	if (debug > 0) logger.log("fft");
	if (debug == 2) xs::NpzSave(npzfile, "dat_preproc", dat.rows(), "w");
	if (debug == 2) logger.log("save dat");

	// KEYGEN //////////////////////////////////////////////////////////////
	auto groups = xs::GroupChannels(chanmap.span());
	auto keys = xs::Arange<uint16_t>(0, stalocs.nrow());
	auto allpairs = xs::UniquePairs(keys);
	// auto pairs = allpairs.rows();
	auto pairs = xs::DistFiltPairs(allpairs.rows(), stalocs.rows(), dist_min, dist_max);
	if (debug > 0) printf("using %lu pairs of %lu total\n", pairs.size(), allpairs.nrow());
	if (debug > 0) logger.log("build sta pairs");
	if (debug == 2) xs::NpzSave(npzfile, "sta_pairs", pairs, "a");
	if (debug == 2) logger.log("save sta pairs");


	// XCORRS //////////////////////////////////////////////////////////////
	auto ccdat = xs::Array2D<float>(pairs.size(), dat.ncol());	
	xs::XCorrCombineChans(fdat, groups, pairs, ccdat, smooth_cc_wlen);
	if (debug > 0) logger.log("xcorr");
	if (debug == 2) xs::NpzSave(npzfile, "dat_cc", ccdat.rows(), "a");
	if (debug == 2) logger.log("save ccfs");	

	// xs::IsValidTTable(pairs, ttable, ccdat.ncol());
	// auto dd = xs::DistDiffPairs(pairs, stalocs.rows());
	// std::cout << "max dist_diff: " << xs::Max(gsl::make_span(dd)) << "\n";

	// SEARCH //////////////////////////////////////////////////////////////
	auto power = xs::Vector<float>(ttable[0].size());
	xs::InterLocBlocks(ccdat.rows(), pairs, ttable, power.span());
	
	// auto power = xs::InterLoc(ccdat.rows(), pairs, ttable);
	// auto power = xs::InterLocBad(ccdat.rows(), pairs, ttable);
	size_t imax = xs::ArgMax(power.span());
	float vmax = xs::Max(power.span());
	if (debug > 0) logger.log("interloc");
	if (debug == 2) xs::NpzSave(npzfile, "grid_power", power.span(), "a");
	if (debug == 2) logger.log("save grid");

	auto rmaxes = xs::RowMaxes(ccdat.rows());
	float max_theor = xs::Mean(gsl::make_span(rmaxes));
	// if (debug > 0) logger.log("cc theor max");	
	float peak_align = vmax / max_theor * 100.0f;
	if (debug > 0) logger.log("cc maxes");
	if (debug > 0) printf("(max_grid / max_theor)= %f / %f = %f%%\n", vmax, max_theor, peak_align);


	// ROLL FOR ORIGIN_TIME ////////////////////////////////////////////////////
	std::vector<uint16_t> wtt;
	for(auto&& row : ttable) wtt.push_back(row[imax]);		
	if (debug > 0) logger.log("tts to source");
	if (debug==2) xs::NpzSave(npzfile, "tts_src", gsl::make_span(wtt), "a");
	if (debug==2) logger.log("save wtt");

	// xs::RollAndStack(dat, groups, wtt.span());
	xs::Envelope(dat.rows());
	if (debug > 0) logger.log("envelope");

	// get origin time by shifting and stacking, len(stack) = 2 * len(sig)
	auto stack = xs::RollAndStack(dat.rows(), groups, gsl::make_span(wtt));	
	int64_t otime = xs::ArgMax(stack.span()) - static_cast<int64_t>(stack.size() / 2);	
	if (debug > 0) logger.log("stack");
	if (debug==2) xs::NpzSave(npzfile, "dat_stack", stack.span(), "a");

	if (debug==2) {
		xs::RollSigs(dat.rows(), groups, gsl::make_span(wtt));
		logger.log("roll");
		xs::NpzSave(npzfile, "dat_rolled", dat.rows(), "a");
	}

	// xs::RollSigs(dat.rows(), groups, gsl::make_span(wtt));
	// if (debug > 0) logger.log("roll");
	// auto stack = xs::StackSigs(dat.rows());
	// if (debug > 0) logger.log("stack");	
	// if (debug==2) xs::NpzSave(npzfile, "dat_rolled", dat.rows(), "a");
	// if (debug==2) xs::NpzSave(npzfile, "dat_stack", stack.span(), "a");

	if (debug > 0) printf("[gmax] %f [imax] %lu [ot_imax] %ld\n", vmax, imax, otime);

	outbuf[0] = static_cast<int64_t>(vmax * 10000); // max power scaled
	outbuf[1] = imax; // tt grid argmax
	outbuf[2] = otime; // origin time argmax

	if (debug > 0) logger.summary();

}

void SearchOnePhase(float* rawdat_p, uint32_t nchan, uint32_t npts, float sr, float* stalocs_p, uint32_t nsta, uint16_t* chanmap_p, std::vector<uint16_t*>& tt_ptrs, uint32_t ngrid, int64_t* outbuf, uint32_t nthreads, int debug, std::string& npzfile) 
{

	omp_set_num_threads(nthreads);
	std::string const HOME = std::getenv("SPP_COMMON");
	std::string file_wisdom = HOME + "fftw3_wisdom.txt";
	fftwf_import_wisdom_from_filename(&file_wisdom[0]);
	// std::cout << "file_wisdom: " << file_wisdom << "\n";

	xs::VecOfSpans<uint16_t> ttable;
	for(auto&& ptr : tt_ptrs) ttable.push_back(gsl::make_span(ptr, ngrid));
	
	auto dat = xs::Array2D<float>(rawdat_p, nchan, npts, true); // copies data
	auto stalocs = xs::Array2D<float>(stalocs_p, nsta, 3, false);
	auto chanmap = xs::Vector<uint16_t>(chanmap_p, nchan);

	xs::WFSearchOnePhase(dat, sr, stalocs, chanmap, ttable, outbuf, debug, npzfile);
	
	fftwf_export_wisdom_to_filename(&file_wisdom[0]);
	
}

}
