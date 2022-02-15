#pragma once

#include <iostream>
#include <string>
#include <assert.h>
#include <numeric>

#include "/usr/include/hdf5/serial/H5Cpp.h"
#include "xseis2/core.h"


namespace xs {


// wrapper for h5 dataset
class H5Dataset {
public:
	H5::DataSet dset_;
	H5::DataSpace filespace_;
	hsize_t nrow_, ncol_, size_;
	size_t rank_;
	H5::DataType dtype_;
	hsize_t shape_[2] = {1, 1};    // dataset dimensions
	
	H5Dataset(H5::DataSet dset): dset_(dset){
		filespace_ = dset_.getSpace();
		rank_ = filespace_.getSimpleExtentNdims();
		// hsize_t shape_[rank];    // dataset dimensions
		rank_ = filespace_.getSimpleExtentDims(shape_);
		assert(rank_ == 1 || rank_ == 2);
		if(rank_ == 1) 
		{
			nrow_ = 1;
			ncol_ = shape_[0];
		}
		else if(rank_ == 2)
		{
			nrow_ = shape_[0];
			ncol_ = shape_[1];
		}
		
		// size_ = shape_[0] * shape_[1];		
		dtype_ = dset_.getDataType();
	}

	template <typename T>	
	Array2D<T> LoadArray(size_t col_offset=0) {

		assert(col_offset < ncol_);
		size_t ncol = ncol_ - col_offset;
		auto arr = Array2D<T>(nrow_, ncol);

		std::vector<size_t> row_keys(nrow_);
		std::iota(std::begin(row_keys), std::end(row_keys), 0);

		LoadRows(arr, row_keys, col_offset);

		return arr;
	}

	template <typename T>	
	Array2D<T> LoadArrayCols(std::vector<size_t> col_range = {0, 0}) {

		size_t col_offset = col_range[0];
		assert(col_offset < ncol_);

		if (col_range[1] == 0) col_range[1] = ncol_;
		// size_t ncol = ncol_ - col_offset;
		auto arr = Array2D<T>(nrow_, col_range[1] - col_range[0]);

		std::vector<size_t> row_keys(nrow_);
		std::iota(std::begin(row_keys), std::end(row_keys), 0);

		LoadRows(arr, row_keys, col_offset);

		return arr;
	}

	template <typename T>	
	void LoadRows(Array2D<T>& arr, std::vector<size_t>& row_keys, size_t col_offset=0) {

		assert(arr.nrow_ == row_keys.size());

		// Define slab size (loading 1 x ncol_ chunks)
		hsize_t count[2] = {1, arr.ncol_};
		// row, col offset to load 
		hsize_t offset[2] = {0, (hsize_t) col_offset};
		H5::DataSpace mspace(rank_, count);

		for(size_t i = 0; i < row_keys.size(); ++i) {
			offset[0] = row_keys[i];
			filespace_.selectHyperslab(H5S_SELECT_SET, count, offset);
			dset_.read(arr.row(i), dtype_, mspace, filespace_);			
		}
	}

	// template <typename T>
	// std::vector<T> LoadVector() {
	// 	std::vector<T> vec(ncol_);
	// 	H5::DataSpace mspace(1, shape_);
	// 	dset_.read(&vec[0], dtype_, mspace, filespace_);
	// 	return vec;
	// }

	template <typename T>
	Vector<T> LoadVector() {
		auto vec = Vector<T>(ncol_);
		H5::DataSpace mspace(1, shape_);
		dset_.read(&vec[0], dtype_, mspace, filespace_);
		return vec;
	}	

};


// wrapper for h5 file
class H5File {
public:
	H5::H5File hfile;

	H5File(const H5std_string file_path){
		hfile = H5::H5File(file_path, H5F_ACC_RDONLY);		
	}

	H5Dataset operator[] (const H5std_string dset_name){		
		return H5Dataset(hfile.openDataSet(dset_name));
	}

	template <typename T>
	T attribute(const H5std_string attr_name){
		T val;
		H5::Attribute attr = hfile.openAttribute(attr_name);
		attr.read(attr.getDataType(), &val);
		return val;
	}

};

} // end namespace xs







