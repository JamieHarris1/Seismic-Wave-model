#!/bin/python3
import numpy as np
import math
import matplotlib.pyplot as plt
from s_wave import SWave
from code.tools import *

# plain simulation
tmax = 5e-1
Np = 401
Xmax = 4000
Zmax = 2000
Xsrc = Xmax/2
nabs = 30
GammaMax = 0.2
# Create instance if seismic wave
sw = SWave(Xmax, Zmax, Np, 0.1, nabs)
sw.excite_f = 20

####################################################
# Single layer of Granite
####################################################
data = []
data.append([2000]+granite)
detector_dx = 100  # distance between detectors
sw.set_model([Xsrc, 0], data, detector_dx)
print("detector_dx =", detector_dx)
# sw.plot_layers("Vp")
# sw.plot_layers("Vs")
# sw.plot_layers("rho")
sw.iterate(tmax, sw.dt, tmax/200)

# We save the detector signals for later
d1 = sw.detectors.copy()

for Xdet in [2000, 2300, 2500]:
	# Prefix for filename with main parameter values
	prefix = "X{}_Z{}_N{}_d{}_f{}_G".format(Xmax, Zmax, Np, int(Xdet-Xsrc),
		sw.excite_f)

	make_detector_signal(sw, Xdet, Xsrc, "v", tmax, "Granite", prefix+"_v.pdf")
	make_detector_signal(sw, Xdet, Xsrc, "w", tmax, "Granite", prefix+"_w.pdf")

####################################################
# Triple layer : Granite Shale Granite
####################################################
data = []
data.append([500]+granite)
data.append([300]+shale)
data.append([1200]+granite)
sw.set_model([Xmax/2, 0], data, detector_dx)
sw.plot_layers("Vp")
sw.plot_layers("Vs")
sw.plot_layers("rho")

sw.iterate(tmax, sw.dt, tmax/200)

# This is the detector signals for the triple layer
d2 = sw.detectors.copy()

# Image snapshots of the v  and w waves.
prefix = "X{}_Z{}_N{}_f{}_na{}_g{}_GSG".format(Xmax, Zmax, Np, sw.excite_f,
	nabs, GammaMax)
sw.snapshots(0, 1e-3, prefix+"_v")
sw.snapshots(1, 1e-3, prefix+"_w")

plt.clf()

for Xdet in [2000, 2300, 2500]:
    # Prefix for filename with main parameter values
    prefix = "X{}_Z{}_N{}_d{}_f{}_GSG".format(Xmax, Zmax, Np, int(Xdet-Xsrc),
    	sw.excite_f)

    make_detector_signal(sw, Xdet, Xsrc, "v", tmax, "Granite-Shale-Granite",
    	prefix+"_v.pdf")
    make_detector_signal(sw, Xdet, Xsrc, "w", tmax, "Granite-Shale-Granite",
    	prefix+"_w.pdf")
    make_detector_signal(sw, Xdet, Xsrc, "Mod", tmax, "Granite-Shale-Granite",
    	prefix+"_Mod.pdf")

    # Plot the detector trace d2 after subtracting the direct signal d1
    make_detector_reduced_signal(sw, d2, d1, Xdet, Xsrc, "v", tmax,
		"Granite-Shale-Granite",
		prefix+"_reduced_v.pdf")
    
    make_detector_reduced_signal(sw, d2, d1, Xdet, Xsrc, "w", tmax,
		"Granite-Shale-Granite",
		prefix+"_reduced_w.pdf")
    
    make_detector_reduced_signal(sw, d2, d1, Xdet, Xsrc, "Mod", tmax,
		"Granite-Shale-Granite",
		prefix+"_reduced_Mod.pdf")
