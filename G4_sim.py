from chroma.event import Vertex, Photons
from mpl_toolkits.mplot3d import Axes3D
from chroma.detector import Detector
from chroma.geometry import Material
from chroma.generator import g4gen
from chroma.sim import Simulation
from chroma.loader import load_bvh
import matplotlib.pyplot as plt
import lensmaterials as lm
import kabamland2 as k2
import numpy as np

def k_thresh(r_idx,m=0.511):
# return the Cerenkov threshold energy (kinetic) for a given particle (mass in MeV) crossing media with r_idx as refractive index
	s = r_idx*r_idx
	return m*(1/np.sqrt(1.0-1.0/s)-1.0)

def sim_test():
	scint = lm.get_scintillation_material()
	gen = g4gen.G4Generator(scint,orb_radius=10.)	
	out_ph1 = gen.generate_photons([Vertex('e-', (0,0,0), (1,0,0), 100)],mute=True)
	#out_ph2 = gen.generate_photons([Vertex('gamma', (0,0,0), (1,0,0), 2)],mute=True)
	#plt.hist(out_ph1.wavelengths,bins=400,histtype='step',linewidth=2,label='e$^-$')
	#plt.hist(out_ph2.wavelengths,bins=400,histtype='step',linewidth=2,label='$\gamma$')
	#plt.xlabel('scintillation photons wavelength [nm]')
	#plt.legend()
	#plt.show()
	#plt.close()
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.plot(out_ph1.pos[:,0],out_ph1.pos[:,1],out_ph1.pos[:,2],'.',label='e$^-$')
	#ax.plot(out_ph2.pos[:,0],out_ph2.pos[:,1],out_ph2.pos[:,2],'.',label='$\gamma$')
	plt.legend()
	plt.show()
	exit()

if __name__ == '__main__':
	sim_test()



'''
def create_gamma_event(location, energy, amount, config, eventname, datadir=""):
	# simulates a number of single gamma photon events equal to amount
	# at position given by location for a given configuration.
	# Gamma energy is in MeV.
	#kabamland = Detector(lm.ls)
	#k2.build_kabamland(kabamland, config)
	# kabamland.add_solid(Solid(make.box(0.1,0.1,0.1,center=location), glass, lm.ls, color=0x0000ff)) # Adds a small blue cube at the event location, for viewing
	#kabamland.flatten()
	#kabamland.bvh = load_bvh(kabamland)
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
'''
