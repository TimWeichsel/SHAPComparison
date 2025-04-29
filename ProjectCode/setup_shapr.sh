#!/bin/sh - execute in shell 

# Clone the shapr git
rm -rf src/external/shapr # remove old shapr git, if exists
git clone https://github.com/NorskRegnesentral/shapr.git src/external/shapr #clone shapr git to source folder
cd src/external/shapr # go to shapr folder
git checkout 9eb3d44 # checkout specific commit
Rscript ../../../setup_r_requirements.R # install R packages
pip install -e python # install shapr package
