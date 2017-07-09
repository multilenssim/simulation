# Originally scripts_stanford.py from Scott Kravitz

from chroma.sim import Simulation
from chroma import sample
from chroma.generator import vertex
from ShortIO.root_short import ShortRootWriter
#import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from chroma.detector import Detector
import lensmaterials as lm

import kabamland2 as kb
import detectoranalysis as da

datadir = "/home/kwells/TestData/"

fileinfo = 'cfJiani3_4'#'configpc6-meniscus6-fl1_027-confined'#'configpc7-pcrad09dia-fl2-confined'#'configview-meniscus6-fl2_113-confined'

#kb.full_detector_simulation(100000, 'cfSam1_1', 'sim-'+fileinfo+'_100million.root', datadir=datadir)

# For an inscribed radius of ~7400, (3000, 3000, 3000) corresponds to a radius of about 70% of the inscribed radius
#kb.create_event((3000,3000,3000), 0.1, 100000, 'cfJiani3_4', 'event-'+fileinfo+'-(3000-3000-3000)-100000.root', datadir=datadir)

kb.create_gamma_event((0,0,0), 1.0, 1, 'cfJiani3_4', 'event-gamma-test-'+fileinfo+'-(0-0-0)-1.root', datadir=datadir)

#kb.create_electron_event((0,0,0), 1.0, 1, 'cfJiani3_4', 'event-electron-test-'+fileinfo+'-(0-0-0)-1.root', datadir=datadir)

#kb.create_event((0, 0, 0), 0.1, 100000, 'cfSam1_1', 'event-'+fileinfo+'-(0-0-0)-100000.root', datadir=datadir)

#da.create_detres('cfJiani3_4', 'sim-'+fileinfo+'_100million.root', 'detresang-'+fileinfo+'_1DVariance_100million.root', method="GaussAngle", nevents=-1, datadir=datadir)
    
#da.check_detres_sigmas('cfJiani3_4', 'detresang-'+fileinfo+'_1DVariance_100million.root', datadir=datadir)

#da.compare_sigmas('cfJiani3_4','detresang-'+fileinfo+'_1DVariance_100million.root','cfJiani3_4','detresang-'+fileinfo+'_1DVariance_noreflect_100million.root',datadir=datadir)

#da.get_AVF_performance('cfJiani3_2', 'event-'+fileinfo+'-(0-0-0)-100000.root', detres='detresang-'+fileinfo+'_1DVariance_100million.root', detbins=20, n_repeat=5, event_pos=[(0.,0.,0.)], n_ph=[100, 1000, 10000], min_tracks=[0.1], chiC=[2.0], temps=[[256,0.25]], debug=False, datadir=datadir)

#da.reconstruct_event_AVF('cfJiani3_2', 'event-'+fileinfo+'-(0-0-0)-100000.root', detres='detresang-'+fileinfo+'_1DVariance_100million.root', event_pos=(0,0,0), detbins=20, chiC=2., min_tracks=0.1, n_ph=100, debug=True, datadir=datadir)
