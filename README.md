# skymap

[![Build](https://img.shields.io/travis/kadrlica/skymap.svg)](https://travis-ci.org/kadrlica/skymap)
[![PyPI](https://img.shields.io/pypi/v/skymap.svg)](https://pypi.python.org/pypi/skymap)
[![Release](https://img.shields.io/github/release/kadrlica/skymap.svg)](../../releases)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](../../)

Python skymaps based on [`matplotlib.basemap`](http://matplotlib.org/basemap/).

# Installation

The best way to install skymap is if you have [anaconda](https://anaconda.org/) installed. First, create a conda environment for skymap:
```
conda create -n skymap numpy scipy pandas matplotlib basemap=1.0.7 astropy ephem healpy nose -c conda-forge
```

Then you can install `skymap` via pip:
```
pip install skymap
```
or if you want the bleeding version, you can clone and install via github:
```
git clone https://github.com/kadrlica/skymap.git
cd skymap
python setup.py install
```
