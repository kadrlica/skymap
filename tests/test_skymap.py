#!/usr/bin/env python
"""
Generic python script.
"""
__author__ = "Alex Drlica-Wagner"

import skymap

def test_create_skymap():
    m = skymap.Skymap()
    m = skymap.McBrydeSkymap()
    m = skymap.OrthoSkymap()
