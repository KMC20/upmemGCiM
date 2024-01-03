cd UPMEM_d
mkdir build
mkdir ckpts
ln -s ../datasets datasets
make run
cd ..
cd UPMEM_h
mkdir build
mkdir ckpts
ln -s ../datasets datasets
make run
cd ..