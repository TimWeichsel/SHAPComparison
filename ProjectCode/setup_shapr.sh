#!/bin/sh - execute in shell 

# Clone the shapr git
Rscript setup_r_requirements.R # install R packes
git clone https://github.com/NorskRegnesentral/shapr.git src/external/shapr #clone shapr git to source folder
cd src/external/shapr # go to shapr folder
git checkout 9eb3d44 # checkout specific commit
Rscript install_r_packages.R # install R package of shapr git
pip install -e python # install shapr package