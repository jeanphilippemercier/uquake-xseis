#include <iostream>
#include <vector>
#include <iterator>
#include <numeric>
#include <string>
#include "xseis2/core.h"
#include "xseis2/logger.h"
#include "xseis2/hdf5.h"
#include "xseis2/npy.h"
#include "xseis2/signal.h"
#include "xseis2/keygen.h"
#include "xseis2/beamform.h"

#include <omp.h>
#include <fftw3.h>


int main(int argc, char const *argv[])
{

	uint32_t nthreads = 4;
	omp_set_num_threads(nthreads);

	auto logger = xseis::Logger();

	// IO ///////////////////////////////////////////////////////////////////////

	std::string const HOME = std::getenv("HOME") ? std::getenv("HOME") : ".";
	std::string dir_dat = HOME + "/data/oyu/synthetic/";
	std::string file_wisdom = "fftw3_wisdom.txt";
	fftwf_import_wisdom_from_filename(&file_wisdom[0]);
	std::string file_out = HOME + "/data/oyu/synthetic/output.npz";
	
	// auto hf = xseis::H5File(dir_dat + "sim_Vp5k.h5");	
	auto hf = xseis::H5File(dir_dat + "sim_p5s3.h5");	
	// auto hf = xseis::H5File(dir_dat + "real_185101.h5");	
	auto sr = static_cast<float>(hf.attribute<double>("samplerate"));
	auto stalocs = hf["sta_locs"].LoadArray<float>();
	auto chanmap = hf["chan_map"].LoadVector<uint16_t>();
	auto dat = hf["data"].LoadArray<float>();
	logger.log("load");

	// PREPROC ///////////////////////////////////////////////////////////////////////
	auto fdat = xseis::WhitenAndFFT(dat, sr, {30, 60, 350, 400});
	// auto fdat = xseis::WhitenAndFFTPadDec2x(dat, sr, {30, 60, 350, 400});
	logger.log("fft");
	xseis::NpzSave(file_out, "sigs_preproc", dat.rows(), "w");
	logger.log("save dat");

	std::cout << "sr: " << sr << "\n";


	// Correlate ///////////////////////////////////////////////////////////////////
	// group similar channels and create cc pairs
	auto keys = xseis::Arange<uint16_t>(0, stalocs.nrow());
	// auto groups = xseis::GroupChannels(keys, chanmap);
	auto groups = xseis::GroupChannels(chanmap.span());

	// for(auto&& i : groups) {
	// 	std::cout << "i: " << i << "\n";
	// }

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
	xseis::XCorrChanGroupsAbs(fdat, groups, pairs, ccdat, 10);
	logger.log("xcorr");
	xseis::NpzSave(file_out, "sigs_xcorrs", ccdat.rows(), "a");
	logger.log("save ccfs");	
	auto rmaxes = xseis::RowMaxes(ccdat.rows());
	auto vmax_cc = xseis::Mean(gsl::make_span(rmaxes));
	logger.log("vmax_cc");


	// 1400, 1300, 1000
	// auto grid = xseis::Grid({1200, 1600, 1100, 1500, 800, 1200, 5});
	// auto grid = xseis::Grid({500, 2000, 500, 1900, 500, 1800, 10});
	auto grid = xseis::Grid({500, 2000, 500, 1900, 200, 1500, 25});
	auto points = grid.points();
	xseis::NpzSave(file_out, "grid_points", points.rows(), "a");
	xseis::NpzSave(file_out, "grid_lims", grid.lims(), "a");
	logger.log("build points");


	float vel = 5000;
	auto ttable = xseis::Array2D<uint16_t>(stalocs.nrow(), points.nrow());
	xseis::FillTravelTimeTable(stalocs, points, vel, sr, ttable);
	logger.log("Fill ttable");
	// xseis::NpzSave(file_out, "ttable", ttable.rows(), "a");
	// logger.log("Save tts");

	auto power = xseis::Vector<float>(points.nrow());
	xseis::InterLocBlocks(ccdat.rows(), pairs, ttable.rows(), power.span());
	logger.log("interloc");
	xseis::NpzSave(file_out, "grid_power", power.span(), "a");

	auto imax = xseis::ArgMax(power.span());
	auto vmax = xseis::Max(power.span());
	std::cout << "wloc: " << points.span(imax) << "\n";
	std::cout << "(vmax/theor): " << vmax << " / " << vmax_cc << " = " << vmax / vmax_cc * 100.0f << "% \n";

	// auto SumRowMax = [=](xseis::VecOfSpans<float> dat){float sum=0; for(auto&& x : dat) sum += xseis::Max(x); return sum / dat.size() * 100;};

	// std::cout << "vmax_cc: " << vmax_cc << "\n";

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
	auto otime = xseis::ArgMax(stack.span());
	std::cout << "otime: " << otime << "\n";
	xseis::NpzSave(file_out, "sigs_rolled", dat.rows(), "a");
	xseis::NpzSave(file_out, "sig_stack", stack.span(), "a");

	logger.summary();
	fftwf_export_wisdom_to_filename(&file_wisdom[0]);


	return 0;
}