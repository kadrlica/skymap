from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

def get_datadir():
    from os.path import abspath,dirname,join
    return join(dirname(abspath(__file__)),'data')

def setdefaults(kwargs,defaults):
    for k,v in defaults.items():
        kwargs.setdefault(k,v)
    return kwargs
