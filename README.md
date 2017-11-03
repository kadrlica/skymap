# skymap

[![Build](https://img.shields.io/travis/kadrlica/skymap.svg)](https://travis-ci.org/kadrlica/skymap)
[![PyPI](https://img.shields.io/pypi/v/skymap.svg)](https://pypi.python.org/pypi/skymap)
[![Release](https://img.shields.io/github/release/kadrlica/skymap.svg)](../../releases)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](../../)

Python skymaps based on [`matplotlib.basemap`](http://matplotlib.org/basemap/).

## Installation

The best way to install skymap is if you have [anaconda](https://anaconda.org/) installed. If you have trouble, check out the [.travis.yml](.travis.yml) file. The procedure below will create a conda environment and pip install skymap:
```
conda create -n skymap numpy scipy pandas matplotlib basemap astropy ephem healpy nose -c conda-forge
source activate skymap
pip install skymap
```
If you want the bleeding edge of skymap, you can follow the directions above to create the conda environment, but then install by cloning directly from github:
```
git clone https://github.com/kadrlica/skymap.git
cd skymap
python setup.py install
```

## Tutorial

If you want to see what you can do with `skymap`, check out the [tutorial](tutorial/).
