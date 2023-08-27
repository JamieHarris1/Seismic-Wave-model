#!/bin/python3
import numpy as np
import math
import matplotlib.pyplot as plt
from s_wave import s_wave
from tools import *

# plain simulation
tmax = 5e-1
Np = 201
Xmax = 2000
Zmax = 1000
Xsrc = Xmax/2
nabs = 15
# Create instance if seismic wave
sw = s_wave(Xmax, Zmax, Np, 0.1, nabs)
sw.excite_f = 20
  
####################################################
# Triple layer : Granite Shale Granite
####################################################
detector_dx = 100 # distance between detectors
data = []
data.append([400]+granite)
data.append([200]+shale)
data.append([400]+granite)
sw.set_model([Xmax/2,0], data, detector_dx)

sw.iterate(tmax, sw.dt, tmax/200)

for Xdet in [1150] :
  # Prefix for filename with main parameter values
  prefix= "X{}_Z{}_N{}_d{}_f{}_GSG".format(Xmax, Zmax, Np, int(Xdet-Xsrc),
                                           sw.excite_f)
     
  make_detector_signal(sw, Xdet, Xsrc, "v", tmax, "Granite-Shale-Granite",
                     prefix+"_v.pdf")
  make_detector_signal(sw, Xdet, Xsrc, "w", tmax, "Granite-Shale-Granite",
                     prefix+"_w.pdf")
  make_detector_signal(sw, Xdet, Xsrc, "Mod", tmax, "Granite-Shale-Granite",
                     prefix+"_Mod.pdf")


