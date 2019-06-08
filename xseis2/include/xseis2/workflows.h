#pragma once

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

// Pre-process, cross-correlate and beamform raw_data using provided traveltime table.
// The provided traveltimes determine which phase (e.g P,S) is searched for
void WFSearchOnePhase(
	Array2D<float>& raw_data, // 2d array of seismic data (nchan x nsamples)
	float samplerate,
	Array2D<float>& station_locations, // each row is station location xyz
	Vector<uint16_t>& channel_map, // maps station locations to raw data channels
	VecOfSpans<uint16_t> ttable, // each row contains traveltime to all grid points for one station
	int64_t* outbuf, // output buffer that results are written to be ready in python
	std::vector<float>& whiten_corner_freqs, // 4 corner frequencies to whiten between
	float pair_dist_min, // uses only correlation pairs between dist_min and dist_max
	float pair_dist_max,
	float cc_smooth_length_sec, // length to absolute value smooth ccfs before beamforming
	int debug_lvl, // debug_lvl level | 0: disabled, 1: logging 2: logging + data dump to debug_file
	std::string& debug_file //
	)
{

	uint32_t const smooth_cc_wlen = cc_smooth_length_sec * samplerate;

	auto logger = xs::Logger();
	if (debug_lvl > 0) logger.log("Start");

	// PREPROC //////////////////////////////////////////////////////////////
	auto zpad_data = xs::ZeroPad(raw_data, raw_data.ncol() * 2);
	if (debug_lvl > 0) logger.log("create padded");

	auto freq_data = xs::FFTAndWhiten(zpad_data, samplerate, whiten_corner_freqs);
	if (debug_lvl > 0) logger.log("fft");
	if (debug_lvl == 2) xs::NpzSave(debug_file, "dat_preproc", zpad_data.rows(), "w");
	if (debug_lvl == 2) logger.log("save dat");

	// KEYGEN //////////////////////////////////////////////////////////////
	auto groups = xs::GroupChannels(channel_map.span());
	auto keys = xs::Arange<uint16_t>(0, station_locations.nrow());
	auto allpairs = xs::UniquePairs(keys);
	auto pairs = xs::DistFiltPairs(allpairs.rows(), station_locations.rows(), pair_dist_min, pair_dist_max);
	if (debug_lvl > 0) printf("using %lu pairs of %lu total\n", pairs.size(), allpairs.nrow());
	if (debug_lvl > 0) logger.log("build sta pairs");
	if (debug_lvl == 2) xs::NpzSave(debug_file, "sta_pairs", pairs, "a");
	if (debug_lvl == 2) logger.log("save sta pairs");


	// XCORRS //////////////////////////////////////////////////////////////
	auto cc_data = xs::Array2D<float>(pairs.size(), zpad_data.ncol());
	xs::XCorrCombineChans(freq_data, groups, pairs, cc_data, smooth_cc_wlen);
	if (debug_lvl > 0) logger.log("xcorr");
	if (debug_lvl == 2) xs::NpzSave(debug_file, "dat_cc", cc_data.rows(), "a");
	if (debug_lvl == 2) logger.log("save ccfs");

	// xs::IsValidTTable(pairs, ttable, cczpad_data.ncol());
	// auto dd = xs::DistDiffPairs(pairs, station_locations.rows());
	// std::cout << "max dist_diff: " << xs::Max(gsl::make_span(dd)) << "\n";

	// SEARCH //////////////////////////////////////////////////////////////
	auto power = xs::Vector<float>(ttable[0].size());
	xs::InterLocBlocks(cc_data.rows(), pairs, ttable, power.span());

	// auto power = xs::InterLoc(cczpad_data.rows(), pairs, ttable);
	// auto power = xs::InterLocBad(cczpad_data.rows(), pairs, ttable);
	size_t argmax = xs::ArgMax(power.span());
	float valmax = xs::Max(power.span());
	if (debug_lvl > 0) logger.log("interloc");
	if (debug_lvl == 2) xs::NpzSave(debug_file, "grid_power", power.span(), "a");
	if (debug_lvl == 2) logger.log("save grid");

	auto rmaxes = xs::RowMaxes(cc_data.rows());
	float max_theor = xs::Mean(gsl::make_span(rmaxes));
	// if (debug_lvl > 0) logger.log("cc theor max");
	float peak_align = valmax / max_theor * 100.0f;
	if (debug_lvl > 0) logger.log("cc maxes");
	if (debug_lvl > 0) printf("(max_grid / max_theor)= %f / %f = %f%%\n", valmax, max_theor, peak_align);


	// ROLL FOR ORIGIN_TIME ////////////////////////////////////////////////////
	std::vector<uint16_t> wtt;
	for(auto&& row : ttable) wtt.push_back(row[argmax]);
	if (debug_lvl > 0) logger.log("tts to source");
	if (debug_lvl==2) xs::NpzSave(debug_file, "tts_src", gsl::make_span(wtt), "a");
	if (debug_lvl==2) logger.log("save wtt");

	// xs::RollAndStack(dat, groups, wtt.span());
	xs::Envelope(zpad_data.rows());
	if (debug_lvl > 0) logger.log("envelope");

	// get origin time by shifting and stacking, len(stack) = 2 * len(sig)
	auto stack = xs::RollAndStack(zpad_data.rows(), groups, gsl::make_span(wtt));
	int64_t otime = xs::ArgMax(stack.span()) - static_cast<int64_t>(stack.size() / 2);
	if (debug_lvl > 0) logger.log("stack");
	if (debug_lvl==2) xs::NpzSave(debug_file, "dat_stack", stack.span(), "a");

	if (debug_lvl==2) {
		xs::RollSigs(zpad_data.rows(), groups, gsl::make_span(wtt));
		logger.log("roll");
		xs::NpzSave(debug_file, "dat_rolled", zpad_data.rows(), "a");
	}

	// xs::RollSigs(zpad_data.rows(), groups, gsl::make_span(wtt));
	// if (debug_lvl > 0) logger.log("roll");
	// auto stack = xs::StackSigs(zpad_data.rows());
	// if (debug_lvl > 0) logger.log("stack");
	// if (debug_lvl==2) xs::NpzSave(debug_file, "dat_rolled", zpad_data.rows(), "a");
	// if (debug_lvl==2) xs::NpzSave(debug_file, "dat_stack", stack.span(), "a");

	if (debug_lvl > 0) printf("[gmax] %f [argmax] %lu [ot_argmax] %ld\n", valmax, argmax, otime);

	outbuf[0] = static_cast<int64_t>(valmax * 10000); // max power scaled
	outbuf[1] = argmax; // tt grid argmax
	outbuf[2] = otime; // origin time argmax

	if (debug_lvl > 0) logger.summary();

}

// Called directly by cython, takes pointers and builds appropriate data structures
void SearchOnePhase(
	float* raw_data_ptr, uint32_t nchan, uint32_t nsample, // ptr to raw data (size = nchan x nsamp)
	float samplerate,
	float* station_locations_ptr, uint32_t nsta, // ptr to station locs (size = nsta x 3)
	uint16_t* channel_map_ptr, // ptr to channel map (size = nchan)
	std::vector<uint16_t*>& ttable_row_ptrs, uint32_t ngrid,
	int64_t* outbuf,
	std::vector<float>& whiten_corner_freqs, // 4 corner frequencies to whiten between
	float pair_dist_min, // uses only correlation pairs between dist_min and dist_max
	float pair_dist_max,
	float cc_smooth_length_sec, // length to absolute value smooth ccfs before beamforming
	uint32_t nthreads,
	int debug_lvl,
	std::string& debug_file
	)
{

	// std::vector<float> whiten_corner_freqs {40, 50, 350, 360};
	// float pair_dist_min = 0;
	// float pair_dist_max = 2000;
	omp_set_num_threads(nthreads);
	std::string const HOME = std::getenv("SPP_COMMON");
	std::string file_wisdom = HOME + "/fftw3_wisdom.txt";
	fftwf_import_wisdom_from_filename(&file_wisdom[0]);
	// std::cout << "file_wisdom: " << file_wisdom << "\n";

	xs::VecOfSpans<uint16_t> ttable;
	for(auto&& ptr : ttable_row_ptrs) ttable.push_back(gsl::make_span(ptr, ngrid));

	auto dat = xs::Array2D<float>(raw_data_ptr, nchan, nsample, true); // copies data
	auto station_locations = xs::Array2D<float>(station_locations_ptr, nsta, 3, false);
	auto channel_map = xs::Vector<uint16_t>(channel_map_ptr, nchan);

	xs::WFSearchOnePhase(dat, samplerate,
		station_locations, channel_map,
		ttable,
		outbuf,
		whiten_corner_freqs,
		pair_dist_min, pair_dist_max,
		cc_smooth_length_sec,
		debug_lvl, debug_file);

	fftwf_export_wisdom_to_filename(&file_wisdom[0]);

}

}
