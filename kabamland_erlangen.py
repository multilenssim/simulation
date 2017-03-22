
from chroma.sim import Simulation
from chroma import sample
from chroma.generator import vertex
from ShortIO.root_short import ShortRootWriter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from chroma.detector import Detector
import lensmaterials as lm
import time 
start = time.time()
import kabamland2 as kb
import detectoranalysis as da

datadir = "/home/exo/"

fileinfo = 'cfJiani3_3'

#kb.full_detector_simulation(100000, 'cfJiani3_3', 'sim-'+fileinfo+'_100million.root', datadir=datadir)

#da.create_detres('cfJiani3_3', 'sim-'+fileinfo+'_100million.root', 'detresang-'+fileinfo+'_noreflect_100million.root', method="GaussAngle", nevents=1, datadir=datadir)
    
da.reconstruct_event_AVF('cfJiani3_3', 'event-cfJiani3_3-(0-0-0)-100000.root', detres='detresang-'+fileinfo+'_noreflect_100million.root', detbins=10, event_pos=(0,0,0), sig_cone=0.01, n_ph=0, min_tracks=4, chiC=3.,  debug=True, datadir=datadir)


#kb.create_event((0,0,0), 0.1, 100000, 'cfJiani3_3', 'event-cfJiani3_3-(0-0-0)-100000.root', datadir)
print time.time()-start, "	sec"
