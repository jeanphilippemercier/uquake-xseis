#ifndef NPY_H
#define NPY_H
// #include <stdint.h>
// #include <iomanip>
// #include <iostream>
// #include <vector>
// #include <stdlib.h>
// #include <malloc.h>
#include "cnpy.h"
#include "xseis2/core.h"

// #define GSL_UNENFORCED_ON_CONTRACT_VIOLATION
// #include "gsl/span"


// typedef std::vector<std::vector<IndexType>> VVui64; 
// typedef std::vector<std::vector<uint16_t>> VVui16; 	

namespace xseis {

template<typename T> void NpySave(std::string fname, Array2D<T>& data) {

	std::vector<size_t> shape {data.nrow_, data.ncol_};
	std::vector<char> header = cnpy::create_npy_header<T>(shape);

	FILE* fp = fopen(fname.c_str(),"wb");

	fseek(fp, 0, SEEK_SET);
	fwrite(&header[0], sizeof(char), header.size(), fp);
	fseek(fp, 0, SEEK_END);

	for(auto&& v : data.rows()) {
		fwrite(&v[0], sizeof(T), shape[1], fp);		
	}

	fclose(fp);
}

template<typename T> void NpySave(std::string fname, gsl::span<T> data, std::string mode = "w") {

	std::vector<size_t> shape;
	shape.push_back(data.size());
	cnpy::npy_save(fname, &data[0], shape, mode);
}

}
#endif
