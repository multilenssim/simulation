import numpy as np
import sys

from chroma.sim import Simulation
from chroma.detector import Detector
from chroma.loader import load_bvh
from chroma.generator import vertex
#from chroma.io.root import RootWriter

import kabamland2 as k2
import lensmaterials as lm

from Geant4.hepunit import *

datadir = "/home/kwells/TestData/"

fileinfo = 'cfJiani3_4'    #'configpc6-meniscus6-fl1_027-confined'#'configpc7-pcrad09dia-fl2-confined'#'configview-meniscus6-fl2_113-confined'


'''
pi = 3.1
hbarc = 4.3
nanometer = 1e-9
ls_refractive_index = 1.5
material = lm.ls

energy = list((2*pi*hbarc / (material.refractive_index[::-1,0] * nanometer)).astype(float))
foo = list([1.0][::,-1,1].astype(float))
'''


##### Scintillation parameters #####  Currenty unused
Scnt_PP = np.array([ 6.6*eV, 6.7*eV, 6.8*eV, 6.9*eV, 7.0*eV, 7.1*eV, 7.2*eV, 7.3*eV, 7.4*eV ])

Scnt_FAST = np.array([ 0.000134, 0.004432, 0.053991, 0.241971, 0.398942, 0.000134, 0.004432, 0.053991, 0.241971 ])
Scnt_SLOW = np.array([ 0.000010, 0.000020, 0.000030, 0.004000, 0.008000, 0.005000, 0.020000, 0.001000, 0.000010 ])

def k_thresh(r_idx,m=0.511):
# return the Cerenkov threshold energy (kinetic) for a given particle (mass in MeV) crossing media with r_idx as refractive index
	s = r_idx*r_idx
	return m*(1/np.sqrt(1.0-1.0/s)-1.0)

def create_scintillaton_material():
	scint = lm.ls
	scint.set('refractive_index', np.linspace(1.5, 1.5, 38))  # Move this to lensmaterial.py.  Set RI to 1.0 to turn off Cherenkov light

	''' Test code
    r_idx = 1.3
    scint = Material('scint_mat')
    scint.set('refractive_index',np.linspace(1.5,1.33,38))
    #scint.set('refractive_index',np.linspace(1.,1.,38))		# Testing inedx 1.0  = air
    scint.density = 1
    scint.composition = { 'H': 1.0/9.0, 'O': 8.0/9.0}
    '''
	energy_ceren = list((2 * pi * hbarc / (scint.refractive_index[::-1, 0].astype(float) * nanometer)))
	energy_scint = list((2 * pi * hbarc / (np.linspace(320, 300, 11).astype(float) * nanometer)))
	spect_scint = list([0.04, 0.07, 0.20, 0.49, 0.84, 1.00, 0.83, 0.55, 0.40, 0.17, 0.03])
	values = list(scint.refractive_index[::-1, 1].astype(float))
	# prop_table.AddProperty('RINDEX', energy_ceren, values)

	# Scintillation properties

	# See https://geant4.web.cern.ch/geant4/UserDocumentation/UsersGuides/ForApplicationDeveloper/html/ch02s03.html
	# Need to validate that the types are being passed through properly.  Previously was using list(Scnt_PP.astype(float)
	scint.set_scintillation_property('FASTCOMPONENT', energy_scint, spect_scint)
	# scint.set_scintillation_property('SLOWCOMPONENT', Scnt_PP, Scnt_SLOW);

	# TODO: These keys much match the Geant4 pmaterial property names.  Get rid of these magic strings.
	scint.set_scintillation_property('SCINTILLATIONYIELD', 20000. / MeV)  # Was 10000 originally
	scint.set_scintillation_property('RESOLUTIONSCALE', 1.0)  # Was 1.0 originally
	scint.set_scintillation_property('FASTTIMECONSTANT', 1. * ns)
	scint.set_scintillation_property('SLOWTIMECONSTANT', 10. * ns)
	scint.set_scintillation_property('YIELDRATIO', 1.0)  # Was 0.8 - I think this is all fast

	# This causes different effects from using the separate FAST and SLOW components below
	#  From KamLAND photocathode paper   # Per Scott
	# kabamland.detector_material.set_scintillation_property('SCINTILLATION', [float(2*pi*hbarc / (360. * nanometer))], [float(1.0)])

	return scint

# Testing an exception hook
# See https://stackoverflow.com/questions/12217537/can-i-force-debugging-python-on-assertionerror
# Didn't work...
def info(type, value, tb):
	if hasattr(sys, 'ps1') or not sys.stderr.isatty() or type != AssertionError:
		# we are in interactive mode or we don't have a tty-like
		# device, so we call the default hook
		sys.__excepthook__(type, value, tb)
	else:
		import traceback, pdb
		# we are NOT in interactive mode, print the exception...
		traceback.print_exception(type, value, tb)
		print
		# ...then start the debugger in post-mortem mode.
		pdb.pm()

sys.excepthook = info

def create_gamma_event(location, energy, amount, config, eventname, datadir=""):
	# simulates a number of single gamma photon events equal to amount
	# at position given by location for a given configuration.
	# Gamma energy is in MeV.
	kabamland = Detector(create_scintillaton_material())
	k2.build_kabamland(kabamland, config)
	# Adds a small blue cube at the event location, for viewing
	# kabamland.add_solid(Solid(make.box(0.1,0.1,0.1,center=location), glass, lm.ls, color=0x0000ff))
	kabamland.flatten()
	kabamland.bvh = load_bvh(kabamland)
	#view(kabamland)
	#f = RootWriter(datadir + eventname)

	sim = Simulation(kabamland, geant4_processes=1)
	print "Starting gun simulation:" + datadir + eventname
	gun = vertex.particle_gun(['gamma'] * amount, vertex.constant(location), vertex.isotropic(),
							  vertex.flat(float(energy) * 0.99, float(energy) * 1.01))
	for ev in sim.simulate(gun, keep_photons_beg=True, keep_photons_end=True, run_daq=False, max_steps=100):
		print 'Photon count: ' + str(ev.nphotons);
		print 'End photons: ' + str(ev.photons_end);
		# f.write_event(ev)
        # f.close()

if __name__ == '__main__':
	create_gamma_event((0, 0, 0), 2., 1, fileinfo, 'gamma-test-', datadir=datadir)


# From Scott's scripts_stanford.py

#kb.full_detector_simulation(100000, 'cfSam1_1', 'sim-'+fileinfo+'_100million.root', datadir=datadir)

# For an inscribed radius of ~7400, (3000, 3000, 3000) corresponds to a radius of about 70% of the inscribed radius
#kb.create_event((3000,3000,3000), 0.1, 100000, 'cfJiani3_4', 'event-'+fileinfo+'-(3000-3000-3000)-100000.root', datadir=datadir)

#kb.create_electron_event((0,0,0), 5.0, 1, 'cfJiani3_4', 'event-electron-test-'+fileinfo+'-(0-0-0)-100000.root', datadir=datadir)

#kb.create_gamma_event((0,0,0), 1.0, 1, 'cfJiani3_4', 'event-gamma-test-'+fileinfo+'-(0-0-0)-1.root', datadir=datadir)

#kb.create_event((0, 0, 0), 0.1, 100000, 'cfSam1_1', 'event-'+fileinfo+'-(0-0-0)-100000.root', datadir=datadir)

#da.create_detres('cfJiani3_4', 'sim-'+fileinfo+'_100million.root', 'detresang-'+fileinfo+'_1DVariance_100million.root', method="GaussAngle", nevents=-1, datadir=datadir)
    
#da.check_detres_sigmas('cfJiani3_4', 'detresang-'+fileinfo+'_1DVariance_100million.root', datadir=datadir)

#da.compare_sigmas('cfJiani3_4','detresang-'+fileinfo+'_1DVariance_100million.root','cfJiani3_4','detresang-'+fileinfo+'_1DVariance_noreflect_100million.root',datadir=datadir)

#da.get_AVF_performance('cfJiani3_2', 'event-'+fileinfo+'-(0-0-0)-100000.root', detres='detresang-'+fileinfo+'_1DVariance_100million.root', detbins=20, n_repeat=5, event_pos=[(0.,0.,0.)], n_ph=[100, 1000, 10000], min_tracks=[0.1], chiC=[2.0], temps=[[256,0.25]], debug=False, datadir=datadir)

#da.reconstruct_event_AVF('cfJiani3_2', 'event-'+fileinfo+'-(0-0-0)-100000.root', detres='detresang-'+fileinfo+'_1DVariance_100million.root', event_pos=(0,0,0), detbins=20, chiC=2., min_tracks=0.1, n_ph=100, debug=True, datadir=datadir)
