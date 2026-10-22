#!/usr/bin/env bash
set -e # exit error
cd "$(dirname "$0")" # go to script folder

# Clone the shapr git
rm -rf src/external/shapr # remove old shapr git, if exists
git clone https://github.com/NorskRegnesentral/shapr.git src/external/shapr #clone shapr git to source folder
cd src/external/shapr # go to shapr folder
git checkout 9eb3d448c595014ba2da756213d1dd834e09cfb0 # checkout specific commit 9eb3d44
Rscript ../../../setup_r_requirements.R # install R packages
pip install -e python # install shapr package
