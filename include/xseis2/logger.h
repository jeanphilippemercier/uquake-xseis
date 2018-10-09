#pragma once

#include <chrono>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>


// #include <iostream>
// #include <iomanip>
// #include <ctime>

// #include <fstream>

namespace xs {

namespace clr {
	const std::string red("\033[0;31m");
	const std::string green("\033[1;32m");
	const std::string yellow("\033[0;33m");
	const std::string cyan("\033[0;36m");
	const std::string magenta("\033[0;35m");
	const std::string reset("\033[0m");
}
// std::cout << "Measured runtime: " << yellow << timer.count() << reset << std::endl;

std::string TimeStamp()
{
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    // std::cout << std::put_time(&tm, "%d-%m-%Y %H-%M-%S") << std::endl;
    std::ostringstream oss;
    oss << std::put_time(&tm, "[%Y-%m-%d %H:%M:%S] ");
    return oss.str();
}

class Logger {

public:
	using clock = std::chrono::system_clock;
	using tunit =  std::chrono::microseconds;
	using stamp = std::pair<std::string, tunit>;

	std::vector<stamp> stamps;
	std::chrono::time_point<clock> t0, tnow;
	std::string color = clr::yellow;
	uint32_t width = 15;

	Logger(){
		start();
		// std::cout << color << TimeStamp() << "Start" << clr::reset << '\n';
	}
	
	void start(){
		t0 = clock::now();
	}

	void log(std::string name){
		tnow = clock::now();
		stamps.push_back(stamp(name, std::chrono::duration_cast<tunit>(tnow - t0)));
		t0 = tnow;
		std::cout << color << TimeStamp() << name << clr::reset << '\n';
	}

	void summary(){
		std::cout.precision(10);
		std::cout << "_____________________________________________\n";
		std::cout << std::left << std::setw(width) << "Name";
		std::cout << std::left << std::setw(width) << "Time (ms)" << "\n";
		std::cout << "_____________________________________________\n";
		double tot = 0;

		for (auto&& stamp: stamps) {
			double elapsed = (double) stamp.second.count() / 1000.;
			std::cout << std::left << std::setw(width) << stamp.first;
			std::cout << std::left << std::setw(width) << elapsed << "\n";
			tot += elapsed;
		}
		std::cout << std::left << std::setw(width) << "TOTAL";
		std::cout << std::left << std::setw(width) << tot << "\n";
	}	
};

}
