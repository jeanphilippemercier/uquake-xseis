#ifndef SIGNAL_H
#define SIGNAL_H

#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>

#include "xseis2/globals.h"
#include "gsl/span"


namespace xseis {


void BuildFreqFilter(const std::vector<float>& corner_freqs, float const sr, gsl::span<float> filter)
{

	uint32_t nfreq = filter.size();
	float fsr = (nfreq * 2 - 1) / sr;
	// printf("nfreq: %u, FSR: %.4f\n", nfreq, fsr);

	std::vector<uint32_t> cx;
	for(auto&& cf : corner_freqs) {
		cx.push_back(static_cast<uint32_t>(cf * fsr + 0.5));
		// printf("cf/fsr %.2f, %.5f\n", cf, fsr);
	}
	// printf("filt corner indexes \n");
	// for(auto&& c : cx) {
	// 	// printf("cx/ cast: %.3f, %u\n", cx, (uint32_t)cx);
	// 	printf("--%u--", c);
	// }
	// printf("\n");

	// whiten corners:  cutmin--porte1---porte2--cutmax

	for(auto& x : filter) {x = 0;}

	// int wlen = porte1 - cutmin;
	float cosm_left = M_PI / (2. * (cx[1] - cx[0]));
	// left hand taper
	for (uint32_t i = cx[0]; i < cx[1]; ++i) {
		filter[i] = std::pow(std::cos((cx[1] - (i + 1) ) * cosm_left), 2.0);
	}

	// setin middle freqs amp = 1
	for (uint32_t i = cx[1]; i < cx[2]; ++i) {
		filter[i] = 1;
	}

	float cosm_right = M_PI / (2. * (cx[3] - cx[2]));

	// right hand taper
	for (uint32_t i = cx[2]; i < cx[3]; ++i) {
		filter[i] = std::pow(std::cos((i - cx[2]) * cosm_right), 2.0);
	}

}

void ApplyFreqFilterReplace(const gsl::span<float> filter, gsl::span<Complex> fdata)
{
	for (uint32_t i = 0; i < filter.size(); ++i)
	{
		if(filter[i] == 0) {
			fdata[i][0] = 0;
			fdata[i][1] = 0;
		}
		else {
			float angle = std::atan2(fdata[i][1], fdata[i][0]);
			fdata[i][0] = filter[i] * std::cos(angle);
			fdata[i][1] = filter[i] * std::sin(angle);
		}		
	}
}


}

#endif



