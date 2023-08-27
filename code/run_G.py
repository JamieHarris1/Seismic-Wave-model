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
detector_dx = 100 # distance between detectors
sw.set_model([Xsrc,0], data, detector_dx)
print("detector_dx =", detector_dx)
#sw.plot_layers("Vp")
#sw.plot_layers("Vs")
#sw.plot_layers("rho")
sw.iterate(tmax, sw.dt, tmax/200)

# We save the detector signals for later
d1 = sw.detectors.copy()

for Xdet in [2000, 2300 ,2500] :
	# Prefix for filename with main parameter values
	prefix= "X{}_Z{}_N{}_d{}_f{}_G".format(Xmax, Zmax, Np, int(Xdet-Xsrc),
										sw.excite_f)

	make_detector_signal(sw, Xdet, Xsrc, "v", tmax, "Granite", prefix+"_v.pdf")
	make_detector_signal(sw, Xdet, Xsrc, "w", tmax, "Granite", prefix+"_w.pdf")
	make_detector_signal(sw, Xdet, Xsrc, "Mod", tmax, "Granite", prefix+"_Mod.pdf")

# Image snapshots of the v  and w waves.  
prefix= "X{}_Z{}_N{}_f{}_na{}_g{}_G".format(Xmax, Zmax, Np, sw.excite_f,
	abs, GammaMax)   
sw.snapshots(0,1e-3,prefix+"_v")
sw.snapshots(1,1e-3,prefix+"_w")
