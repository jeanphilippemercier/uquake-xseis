#include <iostream>
#include <vector>
#include <iterator>
#include <numeric>
// #include "xseis/npy.h"
#include "cnpy.h"
#include <string>
// #include "xseis/structures.h"
// #include "xseis/h5wrap.h"
#include "xseis2/array2d.h"
#include "xseis2/logger.h"
// #include "gsl/span"


int main(int argc, char const *argv[])
{

	auto logger = xseis::Logger();
	// std::vector<float> v(10);
	// std::iota(std::begin(v), std::end(v), 0);
	// std::cout << "descrip: " << v << '\n';
	// cnpy::npy_save("arr1.npy", &v[0], {v.size()}, "w");
	auto dat = xseis::Array2D<float>(3, 5);
	// auto dat = xseis::Array2D<float>({3, 5});

	logger.log("create");

	for(auto& v : dat.rows()) std::iota(std::begin(v), std::end(v), 0);	
	std::cout << "dat: " << dat << '\n';

	logger.log("filled");


	logger.summary();
	
	// std::cout << "dat(0)[6]: " << dat(0)[6] << '\n';

	// std::cout << "dat.row(5): " << dat.row(5) << '\n';

	// std::vector<float, AlignmentAllocator<float, 16> > v (10);
	// std::iota(std::begin(v), std::end(v), 0);

	// std::cout << "v: " << v << '\n';


	// std::cout << "dat(2,3): " << dat(2,3) << '\n';


	// std::cout << "dat(0): " << dat(0) << '\n';
	// // auto span = gsl::make_span(v.begin(), v.end()) ;
	// auto span = gsl::make_span(&v[0], 5);

	// span[3] *= 5;
	// std::cout << "descrip: " << span << '\n';
	// std::cout << "descrip: " << v[12] << '\n';

	// // auto span = gsl::make_span(&v[0], &v[3]);
	// // auto span = gsl::make_span(v);

	// std::cout << "descrip: " << span.size() << '\n';
	// // std::cout << "descrip: " << val << '\n';

	// for(auto&& x : span) {
	// 	std::cout << "descrip: " << x << '\n';
	// }

	return 0;
}