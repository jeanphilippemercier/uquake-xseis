fullfile=$1
filename="${fullfile##*/}"
no_ext="${filename%.*}"
echo running $filename


# gcc -o main msview2.c -lmseed
# g++-6 msview2.cpp -std=c++14 -Wall -o main -lmseed
# g++-6 $fullfile -std=c++14 -Wall -o main -lmseed

# time g++-7 $fullfile -std=c++17 -Wall -o temp -O3 -march=native -ffast-math -lhdf5_serial -lhdf5_cpp -pthread -lfftw3f -lm -fopenmp -lstdc++fs -lcnpy -lz -lbenchmark && time ./temp

# time g++-7 $fullfile -std=c++17 -Wall -o temp -O3 -march=native -ffast-math -lhdf5_serial -lhdf5_cpp -pthread -lfftw3f -lm -fopenmp -lstdc++fs -L../lib/ -lcnpy -lz -lbenchmark && time ./temp

time g++-7 $fullfile -std=c++17 -Wall -o temp -O3 -march=native -ffast-math -lhdf5_serial -lhdf5_cpp -pthread -lfftw3f -lm -fopenmp -lstdc++fs -lcnpy -lz && time ./temp

# time g++-7 $fullfile -std=c++17 -Wall -o temp -O3 -march=native -ffast-math -lhdf5_serial -lhdf5_cpp -pthread -lfftw3f -lm -fopenmp -lmseed -lstdc++fs && time ./temp

# gcc -Wall msview.c -lmseed -o main
# time ./main /home/phil/data/oyu/mseed_new/20180523_185101.mseed
# time ./main /home/phil/data/oyu/mseed_new/20180523_185101_float.mseed
# time ./temp 

# python quickplot3d.py

