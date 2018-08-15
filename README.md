# skymap

[![Build](https://img.shields.io/travis/kadrlica/skymap.svg)](https://travis-ci.org/kadrlica/skymap)
[![PyPI](https://img.shields.io/pypi/v/skymap.svg)](https://pypi.python.org/pypi/skymap)
[![Release](https://img.shields.io/github/release/kadrlica/skymap.svg)](../../releases)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](../../)

The `skymap` package provides a astronomically oriented interface to ploting sky maps based on [`matplotlib.basemap`](http://matplotlib.org/basemap/). This package addresses several issues present in the [`healpy`](https://healpy.readthedocs.io/en/latest/) plotting routines:
1. `healpy` supports a limited set of sky projections (`cartview`, `mollview`, and `gnomview`)
2. `healpy` converts sparse healpix maps to full maps to plot; this is memory intensive for large `nside`

In addition, `skymap` provides some convenience functionality for large optical surveys.

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
