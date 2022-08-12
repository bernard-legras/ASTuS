#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Series of geo plots of the immediate vicinity of the HongaTonga eruption
on 15 January 2022

HongaTonga coordinates:
    20° 33' S
    175° 21' W

This script generates the animation of the first hours after the eruption

Copyright or © or Copr.  Bernard Legras (2022)
under CeCILL-C license "http://www.cecill.info".

bernard.legras@lmd.ipsl.fr
"""

import imageio
images = []

for i in range(36):
    images.append(imageio.imread('Ash/RGB-anim-'+str(i)+'.jpg'))
imageio.mimsave('Ash/movie-near-eruption-hima.gif', images,fps=6)
imageio.mimsave('Ash/movie-near-eruption-hima.mp4', images,fps=6)
