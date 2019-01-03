#ifndef NPY_H
#define NPY_H
// #include <stdint.h>
// #include <iomanip>
// #include <iostream>
// #include <vector>
// #include <stdlib.h>
// #include <malloc.h>
// #include "cnpy/cnpy.h"
#include "cnpy.h"

#include "xseis2/core.h"

// #define GSL_UNENFORCED_ON_CONTRACT_VIOLATION
// #include "gsl/span"


// typedef std::vector<std::vector<IndexType>> VVui64;
// typedef std::vector<std::vector<uint16_t>> VVui16;

namespace xs {

template<typename T> void
NpySave(std::string fname, Array2D<T>& data)
{
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

template<typename T>
void NpySave(std::string fname, gsl::span<T> data, std::string mode = "w")
{
	std::vector<size_t> shape;
	shape.push_back(data.size());
	cnpy::npy_save(fname, &data[0], shape, mode);
}


template<typename T>
void NpzSave(std::string zipname, std::string fname, xs::VecOfSpans<T> data, std::string mode = "w")
{
	using namespace cnpy; // for overloaded + operator

	//first, append a .npy to the fname
	fname += ".npy";

	//now, on with the show
	FILE* fp = NULL;
	uint16_t nrecs = 0;
	size_t global_header_offset = 0;
	std::vector<char> global_header;

	if(mode == "a") fp = fopen(zipname.c_str(),"r+b");

	if(fp) {
		//zip file exists. we need to add a new npy file to it.
		//first read the footer. this gives us the offset and size of the global header
		//then read and store the global header.
		//below, we will write the the new data at the start of the global header then append the global header and footer below it
		size_t global_header_size;
		cnpy::parse_zip_footer(fp,nrecs,global_header_size,global_header_offset);
		fseek(fp,global_header_offset,SEEK_SET);
		global_header.resize(global_header_size);
		size_t res = fread(&global_header[0],sizeof(char),global_header_size,fp);
		if(res != global_header_size){
			throw std::runtime_error("npz_save: header read error while adding to existing zip");
		}
		fseek(fp,global_header_offset,SEEK_SET);
	}
	else {
		fp = fopen(zipname.c_str(),"wb");
	}


	std::vector<size_t> shape;
	size_t nrow = data.size();
	size_t ncol = data[0].size();

	if(data.size() == 1) shape = {ncol};
	else shape = {nrow, ncol};

	// uint32_t nrow = data.size()
	// std::vector<size_t> shape {data.size(), data[0].size()};
	std::vector<char> npy_header = cnpy::create_npy_header<T>(shape);

	size_t nels = std::accumulate(shape.begin(),shape.end(),1,std::multiplies<size_t>());
	size_t nbytes = nels*sizeof(T) + npy_header.size();

	//get the CRC of the data to be added
	uint32_t crc = crc32(0L,(uint8_t*)&npy_header[0],npy_header.size());

	// replace this:	crc = crc32(crc, (uint8_t*) data[0].data(), nels*sizeof(T));
	// with line below to satisfy checksum for non-contig data
	for(auto&& v : data) crc = crc32(crc, (uint8_t*) v.data(), v.size() * sizeof(T));

	//build the local header
	std::vector<char> local_header;
	local_header += "PK"; //first part of sig
	local_header += (uint16_t) 0x0403; //second part of sig
	local_header += (uint16_t) 20; //min version to extract
	local_header += (uint16_t) 0; //general purpose bit flag
	local_header += (uint16_t) 0; //compression method
	local_header += (uint16_t) 0; //file last mod time
	local_header += (uint16_t) 0;     //file last mod date
	local_header += (uint32_t) crc; //crc
	local_header += (uint32_t) nbytes; //compressed size
	local_header += (uint32_t) nbytes; //uncompressed size
	local_header += (uint16_t) fname.size(); //fname length
	local_header += (uint16_t) 0; //extra field length
	local_header += fname;

	//build global header
	global_header += "PK"; //first part of sig
	global_header += (uint16_t) 0x0201; //second part of sig
	global_header += (uint16_t) 20; //version made by
	global_header.insert(global_header.end(),local_header.begin()+4,local_header.begin()+30);
	global_header += (uint16_t) 0; //file comment length
	global_header += (uint16_t) 0; //disk number where file starts
	global_header += (uint16_t) 0; //internal file attributes
	global_header += (uint32_t) 0; //external file attributes
	global_header += (uint32_t) global_header_offset; //relative offset of local file header, since it begins where the global header used to begin
	global_header += fname;

	//build footer
	std::vector<char> footer;
	footer += "PK"; //first part of sig
	footer += (uint16_t) 0x0605; //second part of sig
	footer += (uint16_t) 0; //number of this disk
	footer += (uint16_t) 0; //disk where footer starts
	footer += (uint16_t) (nrecs+1); //number of records on this disk
	footer += (uint16_t) (nrecs+1); //total number of records
	footer += (uint32_t) global_header.size(); //nbytes of global headers
	footer += (uint32_t) (global_header_offset + nbytes + local_header.size()); //offset of start of global headers, since global header now starts after newly written array
	footer += (uint16_t) 0; //zip file comment length

	//write everything
	fwrite(&local_header[0],sizeof(char),local_header.size(),fp);
	fwrite(&npy_header[0],sizeof(char),npy_header.size(),fp);

	// fwrite(data,sizeof(T),nels,fp);
	for(auto&& v : data) {
		fwrite(&v[0], sizeof(T), v.size(), fp);
	}

	fwrite(&global_header[0],sizeof(char),global_header.size(),fp);
	fwrite(&footer[0],sizeof(char),footer.size(),fp);
	fclose(fp);
}


template<typename T>
void NpzSave(std::string zipname, std::string fname, gsl::span<T> data, std::string mode = "w")
{
	xs::VecOfSpans<T> vdata = {data};
	NpzSave(zipname, fname, vdata, mode);
}



}
#endif
