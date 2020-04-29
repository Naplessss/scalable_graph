mkdir GHCN
cd GHCN
wget ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/daily/ghcnd_all.tar.gz
tar -xzvf ghcnd_all.tar.gz
rm ghcnd_all.tar.gz
wget ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt
