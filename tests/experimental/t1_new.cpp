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
	std::string file_out = HOME + "/data/oyu/synthetic/output_new.npz";
	
	auto hf = xseis::H5File(dir_dat + "sim_p5s3.h5");	
	auto sr = hf.attribute<double>("samplerate");
	auto stalocs = hf["sta_locs"].LoadArray<float>();
	auto chanmap = hf["chan_map"].LoadVector<uint16_t>();
	auto dat = hf["data"].LoadArray<float>();
	logger.log("load");

	// PREPROC ///////////////////////////////////////////////////////////////////////

	float wlen_ms = 50.0;
	uint32_t wlen = wlen_ms / 1000.0 * sr;
	// uint32_t flen = wlen / 2 + 1;
	uint32_t taper_len = 5;

	std::vector<float> freq_win {80, 400};
	float fsr = static_cast<float>(wlen) / sr;	
	// printf("nfreq: %lu, FSR: %.4f\n", flen_tot, fsr);
	// size_t fix = static_cast<size_t>(freq_win[0] * fsr + 0.5);
	uint32_t fix_min = freq_win[0] * fsr + 0.5;
	uint32_t fix_max = freq_win[1] * fsr + 0.5;
	uint32_t flen = fix_max - fix_min;
	uint32_t flen_pad = xseis::PadToBytes<xseis::Complex32>(flen, xseis::CACHE_LINE);

	std::cout << "flen: " << flen << "\n";
	std::cout << "flen_pad: " << flen_pad << "\n";
	std::cout << "wlen: " << wlen << "\n";
	std::cout << "fix_min: " << fix_min << "\n";

	// uint32_t blockstep = xseis::PadToBytes<float>(wlen / 2, xseis::MEM_ALIGNMENT);
	uint16_t blockstep = wlen / 2;
	auto iblocks = xseis::Arange<uint32_t>(0, dat.ncol(), blockstep);
	xseis::NpzSave(file_out, "iblocks", gsl::make_span(iblocks), "w");

	std::cout << "blockstep: " << blockstep << "\n";
	std::cout << "iblocks: " << iblocks << "\n";
	// assert(iblocks[2] * sizeof(float) % xseis::MEM_ALIGNMENT == 0);

	auto bdat = xseis::Array2D<float>(dat.nrow(), iblocks.size() * wlen);
	auto fbdat = xseis::Array2D<xseis::Complex32>(dat.nrow(), iblocks.size() * flen_pad);

	auto tmp = xseis::Vector<float>(wlen);
	auto ftmp = xseis::Vector<xseis::Complex32>(wlen / 2 + 1);
	
	fftwf_plan plan_fwd = fftwf_plan_dft_r2c_1d(wlen, tmp.data(), reinterpret_cast<fftwf_complex*>(ftmp.data()), xseis::FFTW_PATIENCE);

	fftwf_plan plan_inv = fftwf_plan_dft_c2r_1d(wlen, reinterpret_cast<fftwf_complex*>(ftmp.data()), tmp.data(), xseis::FFTW_PATIENCE);


	logger.log("misc");
	// check blocks /////////////////////////////////////////////////////////////
	for(size_t irow = 0; irow < dat.nrow(); ++irow) {
		xseis::Fill(bdat.span(irow), 0.0f);

		for(size_t ib = 0; ib < iblocks.size(); ++ib) {
			auto in = gsl::make_span(dat.row(irow) + iblocks[ib], wlen);
			xseis::Copy(in, tmp.span());
			xseis::TaperCosine(tmp.span(), taper_len);
			auto out = gsl::make_span(bdat.row(irow) + ib * wlen, wlen);			
			xseis::Copy(tmp.span(), out);
		}
	}

	xseis::NpzSave(file_out, "bdat", bdat.rows(), "a");
	logger.log("prepare blocks");

	// check blocks /////////////////////////////////////////////////////////////
	for(size_t irow = 0; irow < dat.nrow(); ++irow) {

		for(size_t ib = 0; ib < iblocks.size(); ++ib) {
			auto in = gsl::make_span(dat.row(irow) + iblocks[ib], wlen);
			xseis::Copy(in, tmp.span());
			xseis::TaperCosine(tmp.span(), taper_len);
			fftwf_execute(plan_fwd);
			auto fblk = fbdat.row(irow) + ib * flen_pad;
			xseis::Copy(ftmp.data() + fix_min, flen, fblk);
			// xseis::Whiten(fblk, flen);

			auto out = gsl::make_span(bdat.row(irow) + ib * wlen, wlen);			
			xseis::Fill(ftmp.span(), {0.0f, 0.0f});
			xseis::Whiten(fblk, flen);
			xseis::Copy(fblk, flen, ftmp.data() + fix_min);
			fftwf_execute(plan_inv);
			xseis::Copy(tmp.span(), out);
			xseis::Multiply(out, 1.0f / wlen);
			// float eg = process::Energy(ptr_fw, flen);
			// process::Multiply(ptr_fw, flen, 1.0 / eg);
		}
	}

	logger.log("prepare blocks");
	xseis::NpzSave(file_out, "bdat2", bdat.rows(), "a");

	// Pairs ///////////////////////////////////////////////////////////////////
	// group similar channels and create cc pairs
	auto keys = xseis::Arange<uint16_t>(0, stalocs.nrow());
	auto groups = xseis::GroupChannels(chanmap.span());
	auto pairs_all = xseis::UniquePairs(keys);
	auto pairs = pairs_all.rows();
	// auto pairs = xseis::DistFiltPairs(pairs_all.rows(), stalocs.rows(), 300, 1500);
	xseis::NpzSave(file_out, "sta_ckeys", pairs_all.rows(), "a");
	std::cout << "npairs: " << pairs.size() << '\n';
	logger.log("keygen");

	// Grid ///////////////////////////////////////////////////////////////////
	// src_loc = 1600, 1400, 1000
	auto grid = xseis::Grid({500, 2000, 500, 1900, 200, 1500, 25});
	auto points = grid.points();
	xseis::NpzSave(file_out, "grid_points", points.rows(), "a");
	xseis::NpzSave(file_out, "grid_lims", grid.lims(), "a");
	logger.log("build points");

	float vel = 5000;
	// float vel = 3000;
	auto ttable = xseis::Array2D<uint16_t>(points.nrow(), stalocs.nrow());
	// xseis::FillTravelTimeTable(stalocs, points, vel, sr, ttable);
	xseis::FillTravelTimeTable(points, stalocs, vel, sr, ttable);
	logger.log("Fill ttable");
	// xseis::NpzSave(file_out, "ttable", ttable.rows(), "a");

	uint32_t ngrid = ttable.nrow();
	uint32_t nsta = ttable.ncol();
	uint32_t nchan = fbdat.nrow();
	// std::vector<uint16_t*> dptrs(nsta);
	std::vector<uint16_t> offsets(nsta);
	// xseis::VecOfSpans<xseis::Complex32> dwins(nchan);

	// for(size_t gix = 0; gix < ngrid; ++gix) {
	uint16_t origintime = 266;
	size_t gix = 130032;

	std::vector<uint16_t> wblocks;

	auto ttp = ttable.row(gix);
	for(size_t ista = 0; ista < nsta; ++ista) {
		uint16_t iblock = (origintime + ttp[ista]) / blockstep;
		wblocks.push_back(iblock);
		offsets[ista] = iblock * flen_pad;		
	}
	xseis::NpzSave(file_out, "wblocks", gsl::make_span(wblocks), "a");

	// for(size_t i = 0; i < pairs.size(); ++i) {
	// 	// auto csig = ccdat.span(i);
	// 	// Fill(csig, 0.0f);

	// 	uint32_t nstack = 0;
	// 	auto pair = pairs[i];
	// 	for(auto&& k0 : groups[pair[0]]) {
	// 		for(auto&& k1 : groups[pair[1]]) {
	// 			XCorr(fbdat.row(k0), fbdat.row(k1), fbuf.data(), fbuf.size());
	// 			Convolve(&vshift[0], fbuf.data(), fbuf.size());
	// 			fftwf_execute_dft_c2r(plan_inv, fptr, buf.data());
	// 			Multiply(buf.span(), 1.0f / energy);
	// 			for(size_t j=0; j < buf.size(); ++j) csig[j] += std::abs(buf[j]);	
	// 			nstack++;
	// 		}
	// 	}
	// 	// Multiply(csig, wlen, 1.0f / static_cast<float>(nstack));		
	// 	Multiply(csig, 1.0f / static_cast<float>(nstack));
	// 	// uint32_t zlen = 10;
	// 	// for(size_t k = wlen / 2 - zlen; k < wlen / 2 + zlen; ++k) csig[k] = 0;
	// 	Copy(csig, buf.span());
	// 	if(wlen_smooth != 0) SlidingWinMax(buf.span(), csig, wlen_smooth);
	// }

	// }

// uint32_t tt = ot + tt_ixs[ichan];
// 					uint32_t iblock = tt / dt;			
// 					uint32_t rollby = tt % dt;

	// return 0;

	// auto ccdat = xseis::Array2D<float>(pairs.size(), dat.ncol());	
	// // xseis::XCorrChanGroupsEnvelope(fdat, groups, pairs, ccdat);
	// xseis::XCorrChanGroupsAbs(fdat, groups, pairs, ccdat);
	// logger.log("xcorr");
	// xseis::NpzSave(file_out, "sigs_xcorrs", ccdat.rows(), "a");
	// logger.log("save ccfs");	
	// auto rmaxes = xseis::RowMaxes(ccdat.rows());
	// auto vmax_cc = xseis::Mean(gsl::make_span(rmaxes));
	// logger.log("vmax_cc");



	
	// auto power = xseis::Vector<float>(points.nrow());
	// xseis::InterLocBlocks(ccdat.rows(), pairs, ttable.rows(), power.span());
	// logger.log("interloc");
	// xseis::NpzSave(file_out, "grid_power", power.span(), "a");

	// auto imax = xseis::ArgMax(power.span());
	// auto vmax = xseis::Max(power.span());
	// std::cout << "wloc: " << points.span(imax) << "\n";
	// std::cout << "(vmax/theor): " << vmax << " / " << vmax_cc << " = " << vmax / vmax_cc * 100.0f << "% \n";

	// // auto SumRowMax = [=](xseis::VecOfSpans<float> dat){float sum=0; for(auto&& x : dat) sum += xseis::Max(x); return sum / dat.size() * 100;};

	// // std::cout << "vmax_cc: " << vmax_cc << "\n";

	// auto wtt = ttable.copy_col(imax);
	// xseis::NpzSave(file_out, "tts_to_max", wtt.span(), "a");
	// logger.log("wtt");

	// // xseis::NpySave(dir_dat + "d1.npy", dat);
	// // xseis::RollAndStack(dat, groups, wtt.span());
	// xseis::Envelope(dat.rows());
	// logger.log("envelope");	
	// xseis::RollSigs(dat.rows(), groups, wtt.span());
	// logger.log("roll");
	// auto stack = xseis::StackSigs(dat.rows());
	// logger.log("stack");
	// auto otime = xseis::ArgMax(stack.span());
	// std::cout << "otime: " << otime << "\n";
	// xseis::NpzSave(file_out, "sigs_rolled", dat.rows(), "a");
	// xseis::NpzSave(file_out, "sig_stack", stack.span(), "a");

	logger.summary();
	fftwf_export_wisdom_to_filename(&file_wisdom[0]);


	return 0;
}