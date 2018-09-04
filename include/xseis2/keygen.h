#pragma once

#include "xseis2/core.h"
#include "xseis2/beamform.h"
// #include "xseis2/signal.h"
// #include <map>

// #include "xseis/utils.h"
// #include <random>


namespace xseis {

// using KeyPairs = std::vector<uint16_t[2]>;

size_t NChoose2(size_t n)
{
	return (n * (n-1)) / 2;
}


KeyGroups GroupChannels(std::vector<uint16_t>& keys, Vector<uint16_t>& chanmap)
{

	KeyGroups groups;
	size_t nkeys = keys.size();

	for(auto&& k : keys) {

		std::vector<uint16_t> group;
		
		for(size_t j = 0; j < chanmap.size_; ++j) {
			if (k == chanmap[j]) {
				group.push_back(j);				
			}			
		}
		groups.push_back(group);
	}		
	return groups;
}

Array2D<uint16_t> UniquePairs(std::vector<uint16_t>& keys)
{	
	uint32_t nsig = keys.size();
	auto pairs = Array2D<uint16_t>(NChoose2(nsig), 2);
	size_t row_ix = 0;

	for (uint16_t i = 0; i < nsig; ++i)
	{
		for (uint16_t j = i + 1; j < nsig; ++j)
		{
			pairs(row_ix, 0) = i;
			pairs(row_ix, 1) = j;
			row_ix += 1;
		}
	}
	return pairs;
}

VecOfSpans<uint16_t> DistFiltPairs(VecOfSpans<uint16_t> pairs, VecOfSpans<float> locs, float dmin, float dmax=999999)
{		
	VecOfSpans<uint16_t> pairs_filt;

	for(auto&& p : pairs) 
	{
		float dist = DistCartesian(locs[p[0]], locs[p[1]]);
		if (dist > dmin && dist < dmax) pairs_filt.push_back(p);		
	}

	return pairs_filt;
}



// KeyPairs UniquePairs(std::vector<uint16_t>& keys)
// {
// 	uint32_t nsig = keys.size();
// 	KeyPairs pairs;
// 	pairs.reserve(NChoose2(nsig));
	
// 	size_t row_ix = 0;

// 	for (uint32_t i = 0; i < nsig; ++i)
// 	{
// 		for (uint32_t j = i + 1; j < nsig; ++j)
// 		{
// 			pairs[row_ix][0] = i;
// 			pairs[row_ix][1] = j;
// 			row_ix += 1;
// 		}
// 	}

// 	return pairs;
// }


// KeyPairs DistFilt(std::vector<uint16_t>& keys, Array2D<float>& stalocs, float min_dist, float max_dist)
// {
// 	// size_t npair = 0;

// 	size_t nkeys = keys.size();
// 	// size_t npair_max = NChoose2(nkeys);
// 	std::vector<uint16_t> pairs_flat;
// 	// size_t row_ix = 0;
// 	float dist;
// 	float* loc1 = nullptr;
// 	float* loc2 = nullptr;
	
// 	for (uint32_t i = 0; i < nkeys; ++i)
// 	{
// 		loc1 = stalocs.row(keys[i]);

// 		for (uint32_t j = i + 1; j < nkeys; ++j)
// 		{
// 			loc2 = stalocs.row(keys[j]);
// 			dist = DistCartesian(loc1, loc2);

// 			if (dist > min_dist && dist < max_dist)
// 			{
// 				pairs_flat.push_back(keys[i]);
// 				pairs_flat.push_back(keys[j]);				
// 			}			
// 		}
// 	}
// 	auto pairs = Array2D<uint16_t>(pairs_flat, 2);

// 	return pairs;
// }




// KeyPairs DistFilt(std::vector<uint16_t>& keys, Array2D<float>& stalocs, float min_dist, float max_dist)
// {
// 	// size_t npair = 0;

// 	size_t nkeys = keys.size();
// 	// size_t npair_max = NChoose2(nkeys);
// 	std::vector<uint16_t> pairs_flat;
// 	// size_t row_ix = 0;
// 	float dist;
// 	float* loc1 = nullptr;
// 	float* loc2 = nullptr;
	
// 	for (uint32_t i = 0; i < nkeys; ++i)
// 	{
// 		loc1 = stalocs.row(keys[i]);

// 		for (uint32_t j = i + 1; j < nkeys; ++j)
// 		{
// 			loc2 = stalocs.row(keys[j]);
// 			dist = DistCartesian(loc1, loc2);

// 			if (dist > min_dist && dist < max_dist)
// 			{
// 				pairs_flat.push_back(keys[i]);
// 				pairs_flat.push_back(keys[j]);				
// 			}			
// 		}
// 	}
// 	auto pairs = Array2D<uint16_t>(pairs_flat, 2);

// 	return pairs;
// }


}

