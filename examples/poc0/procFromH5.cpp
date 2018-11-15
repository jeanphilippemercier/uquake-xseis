#include <iostream>
#include <vector>
#include <iterator>
#include <numeric>
#include <string>
#include "xseis2/core.h"
#include "xseis2/hdf5.h"
#include "xseis2/logger.h"
#include "xseis2/workflows.h"

// #include "xseis2/npy.h"
// #include "xseis2/signal.h"
// #include "xseis2/keygen.h"
// #include "xseis2/beamform.h"

#include <omp.h>
// #include <fftw3.h>


int main(int argc, char const *argv[])
{

	uint32_t nthreads = 4;	
	omp_set_num_threads(nthreads);
	int debug = 2;

	auto logger = xs::Logger();

	// IO ///////////////////////////////////////////////////////////////////////
	std::string const HOME = std::getenv("SPP_COMMON");
	std::string dir_dat = HOME + "synthetic/";

	// std::string const XSHOME = std::getenv("SPP_COMMON");
	std::string file_wisdom = HOME + "fftw3_wisdom.txt";
	fftwf_import_wisdom_from_filename(&file_wisdom[0]);
	std::cout << "file_wisdom: " << file_wisdom << "\n";

	std::string file_out = HOME + "output.npz";
	
	auto hf = xs::H5File(dir_dat + "sim_nll.h5");	
	// auto hf = xs::H5File(dir_dat + "sim_dat3.h5");	
	// auto hf = xs::H5File(dir_dat + "sim_nll_noise.h5");	
	// auto hf = xs::H5File(dir_dat + "real_185101.h5");	
	auto sr = static_cast<float>(hf.attribute<double>("samplerate"));
	auto stalocs = hf["sta_locs"].LoadArray<float>();
	auto chanmap = hf["chan_map"].LoadVector<uint16_t>();
	auto dat = hf["data"].LoadArray<float>();
	logger.log("load");

	auto hf_tt = xs::H5File(dir_dat + "nll_ttable.h5");	
	auto ttable = hf_tt["tts_p"].LoadArray<uint16_t>();	
	// auto ttable = hf_tt["tts_s"].LoadArray<uint16_t>();	
	logger.log("load_tt");
	// auto points = hf_tt["grid_locs"].LoadArray<float>();	
	// logger.log("load_glocs");

	std::vector<int64_t> outbuf(3);

	xs::WFSearchOnePhase(dat, sr, stalocs, chanmap, ttable.rows(), outbuf.data(), debug, file_out);

	logger.log("FULLSEARCH");
	logger.summary();

	fftwf_export_wisdom_to_filename(&file_wisdom[0]);

	return 0;
}