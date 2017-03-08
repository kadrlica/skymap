import sys
import os
try: from setuptools import setup
except ImportError: from distutils.core import setup
import versioneer

here = os.path.abspath(os.path.dirname(__file__))

def read(filename):
    return open(os.path.join(here,filename)).read()

setup(
    name='skymap',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    url='https://github.com/kadrlica/skymap',
    author='Alex Drlica-Wagner',
    author_email='kadrlica@fnal.gov',
    scripts = [],
    install_requires=[
        'matplotlib',
        'numpy',
        'scipy',
        'basemap'
    ],
    packages=['skymap'],
    description="A place to hold python tools.",
    long_description=read('README.md'),
    platforms='any',
    keywords='python tools',
    classifiers = [
        'Programming Language :: Python',
        'Development Status :: 2 - Pre-Alpha',
        'Natural Language :: English',
        'Intended Audience :: Science/Research',
    ]
)
