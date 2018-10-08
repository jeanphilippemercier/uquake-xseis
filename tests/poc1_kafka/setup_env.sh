############################################################
# Python
############################################################
sudo apt install python3-pip
sudo apt-get install python3-tk
pip3 install ipython
pip3 install obspy
pip3 install cython
pip3 install kafka


############################################################
# XSEIS
############################################################
sudo apt install libhdf5-dev -y
sudo apt-get install libfftw3-dev libfftw3-doc -y

sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
sudo apt-get update
sudo apt-get install gcc-7 g++-7 -y

mkdir ~/projects
cd ~/projects
git clone git@gitlab.com:pdales/xseis.git

cat >> ~/.profile <<EOF
##############################################
# XSEIS 
###############################################
export XSHOME="${HOME}/projects/xseis"
export PYTHONPATH="${PYTHONPATH}:${XSHOME}/pyinclude"
export CPLUS_INCLUDE_PATH="${CPLUS_INCLUDE_PATH}:${XSHOME}/include"
export LD_LIBRARY_PATH="${XSHOME}/lib:$LD_LIBRARY_PATH"
export LIBRARY_PATH="${XSHOME}/lib:$LIBRARY_PATH"
export CPLUS_INCLUDE_PATH="${CPLUS_INCLUDE_PATH}:/usr/include/hdf5/serial/"
##############################################
EOF

source ~/.profile

# compile the cython portion of xseis
cd xseis/cython
bash compile.sh







# # NOT SURE ABOUT ANYTHING BELOW THIS LINE ###########################################################

# ############################################################
# # microquake and SPP
# ############################################################
# git clone http://seismic-gitlab.eastus.cloudapp.azure.com/root/microquake.git
# git clone http://seismic-gitlab.eastus.cloudapp.azure.com/root/seismic-processing-platform.git


# # Add Kafka Servers into hosts file 
# sudo su -c "<<EOF echo '
# 10.0.0.13 kafka-node-001
# 10.0.0.14 kafka-node-002
# 10.0.0.15 kafka-node-003
# ' >> /etc/hosts
# EOF"


# # Add Environment variables in profile

# cat >> ~/.profile <<EOF
# #######
# #### Seismic Project Variables
# export SPP_CONFIG="/home/spadmin/projects/seismic-processing-platform/config"
# export SPP_COMMON="/home/spadmin/projects/seismic-processing-platform/common"
# ####

# # MTH: add path to NLLOC executables:
# export PATH=$PATH:/home/spadmin/projects/nlloc/bin

# export PYTHONPATH="${PYTHONPATH}:${HOME}/projects/seismic-processing-platform"


# ### Fix pip
# export LC_ALL=C

# # first create the directory 
# export SHARED_DIR=/mnt/seismic_shared_storage
# sudo mkdir -p $SHARED_DIR

# # mount the shared cloud FS on the directory
# sudo mount -t cifs //spsharedstorageaccount.file.core.windows.net/seismic-shared-storage $SHARED_DIR -o vers=3.0,username=spsharedstorageaccount,password=uB0tk++iIIb3A/VRORQ7453wD974qyXENxrtmM/meZeBqBmH9f1ZGyikKymbRhaV0ulVbq25GAgIwL1C3ydmiQ==,dir_mode=0777,file_mode=0777,sec=ntlmssp
# #######
# EOF


# # Reload .profile
# source ~/.profile