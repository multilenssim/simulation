from chroma.event import Vertex, Photons
from mpl_toolkits.mplot3d import Axes3D
from chroma.detector import Detector
from chroma.geometry import Material
from chroma.generator import g4gen
from chroma.sim import Simulation
from chroma.loader import load_bvh
from chroma.generator import vertex
import matplotlib.pyplot as plt
import lensmaterials as lm
import kabamland2 as k2
import numpy as np

import Geant4
from Geant4.hepunit import *

##### Scintillation parameters #####
Scnt_PP = np.array([ 6.6*eV, 6.7*eV, 6.8*eV, 6.9*eV, 7.0*eV, 7.1*eV, 7.2*eV, 7.3*eV, 7.4*eV ])

Scnt_FAST = np.array([ 0.000134, 0.004432, 0.053991, 0.241971, 0.398942, 0.000134, 0.004432, 0.053991, 0.241971 ])
Scnt_SLOW = np.array([ 0.000010, 0.000020, 0.000030, 0.004000, 0.008000, 0.005000, 0.020000, 0.001000, 0.000010 ])


def k_thresh(r_idx,m=0.511):
# return the Cerenkov threshold energy (kinetic) for a given particle (mass in MeV) crossing media with r_idx as refractive index
	s = r_idx*r_idx
	return m*(1/np.sqrt(1.0-1.0/s)-1.0)

def sim_test():
	r_idx = 1.3
	scint = Material('scint_mat')
	scint.set('refractive_index',np.linspace(1.5,1.33,38))
	scint.density = 1
	scint.composition = { 'H': 1.0/9.0, 'O': 8.0/9.0}

	energy_ceren = list((2*pi*hbarc/(scint.refractive_index[::-1, 0].astype(float)*nanometer)))
	energy_scint = list((2*pi*hbarc/(np.linspace(320,300,11).astype(float)*nanometer)))
	spect_scint = list([0.04, 0.07, 0.20, 0.49, 0.84, 1.00, 0.83, 0.55, 0.40, 0.17, 0.03])
	values = list(scint.refractive_index[::-1, 1].astype(float))
	#prop_table.AddProperty('RINDEX', energy_ceren, values)
	#prop_table.AddProperty('FASTCOMPONENT',energy_scint,spect_scint)
	scint.set_scintillation_property('FASTCOMPONENT', energy_scint, spect_scint);


	# Scintillation properties
	# TODO: These keys much match the Geant4 pmaterial property names.  Get rid of these magic strings.
	scint.set_scintillation_property('SCINTILLATIONYIELD', 20000. / MeV)    # Was 10000 originally
	scint.set_scintillation_property('RESOLUTIONSCALE', 1.0)      # Was 1.0 originally
	scint.set_scintillation_property('FASTTIMECONSTANT', 1. * ns)
	scint.set_scintillation_property('SLOWTIMECONSTANT', 10. * ns)
	scint.set_scintillation_property('YIELDRATIO', 1.0)  # Was 0.8 - I think this is all fast

	print("Starting photon generation")
	gen = g4gen.G4Generator(scint)	
	out_ph1 = gen.generate_photons([Vertex('e-', (0,0,0), (1,0,0), 2)],mute=False)
	out_ph2 = gen.generate_photons([Vertex('gamma', (0,0,0), (1,0,0), 2)],mute=False)

	print("e- photons: " + str(len(out_ph1.pos)));
	print("gamma photons: " + str(len(out_ph2.pos)));


	plt.hist(out_ph1.wavelengths,bins=400,histtype='step',linewidth=2,label='e$^-$')
	plt.hist(out_ph2.wavelengths,bins=400,histtype='step',linewidth=2,label='$\gamma$')
	plt.xlabel('scintillation photons wavelength [nm]')
	plt.legend()
	plt.show()
	plt.close()
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.plot(out_ph1.pos[:,0],out_ph1.pos[:,1],out_ph1.pos[:,2],'.',label='e$^-$')
	ax.plot(out_ph2.pos[:,0],out_ph2.pos[:,1],out_ph2.pos[:,2],'.',label='$\gamma$')
	plt.legend()
	plt.show()
	exit()

#Geant4.gApplyUICommand("/run/verbose 2")
#Geant4.gApplyUICommand("/event/verbose 2")
#Geant4.gApplyUICommand("/tracking/verbose 2")

if __name__ == '__main__':
	sim_test()

def create_gamma_event(location, energy, amount, config, eventname, datadir=""):
	# simulates a number of single gamma photon events equal to amount
	# at position given by location for a given configuration.
	# Gamma energy is in MeV.
	kabamland = Detector(lm.ls)
	k2.build_kabamland(kabamland, config)
	# kabamland.add_solid(Solid(make.box(0.1,0.1,0.1,center=location), glass, lm.ls, color=0x0000ff)) # Adds a small blue cube at the event location, for viewing
	kabamland.flatten()
	kabamland.bvh = load_bvh(kabamland)
	#view(kabamland)
	# quit()
	# f = RootWriter(datadir + eventname)

	# Scintillation properties
	# TODO: These keys much match the Geant4 pmaterial property names.  Get rid of these magic strings.
	kabamland.detector_material.set_scintillation_property('SCINTILLATIONYIELD', 10000. / MeV)
	kabamland.detector_material.set_scintillation_property('RESOLUTIONSCALE', 0.0)
	kabamland.detector_material.set_scintillation_property('FASTTIMECONSTANT', 1. * ns)
	kabamland.detector_material.set_scintillation_property('SLOWTIMECONSTANT', 10. * ns)
	kabamland.detector_material.set_scintillation_property('YIELDRATIO', 0.8)

	# This causes different effects from using the separate FAST and SLOW components below
	# kabamland.detector_material.set_scintillation_property('SCINTILLATION', [float(2*pi*hbarc / (360. * nanometer))], [float(1.0)]) # From KamLAND photocathode paper   # Per Scott

	# See https://geant4.web.cern.ch/geant4/UserDocumentation/UsersGuides/ForApplicationDeveloper/html/ch02s03.html
	# Need to validate that the types are being passed through properly.  Previously was using list(Scnt_PP.astype(float)
	kabamland.detector_material.set_scintillation_property('FASTCOMPONENT', Scnt_PP, Scnt_FAST);
	kabamland.detector_material.set_scintillation_property('SLOWCOMPONENT', Scnt_PP, Scnt_SLOW);

	sim = Simulation(kabamland, geant4_processes=1)
	print "Starting gun simulation:" + datadir + eventname
	gun = vertex.particle_gun(['gamma'] * amount, vertex.constant(location), vertex.isotropic(),
							  vertex.flat(float(energy) * 0.99, float(energy) * 1.01))
	for ev in sim.simulate(gun, keep_photons_beg=True, keep_photons_end=True, run_daq=False, max_steps=100):
		print 'Photon count: ' + str(ev.nphotons);
		print 'End photons: ' + str(ev.photons_end);
		# f.write_event(ev)
        # f.close()

