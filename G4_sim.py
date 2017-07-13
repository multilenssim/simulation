from chroma.detector import Detector
from chroma.generator import vertex
from chroma.sim import Simulation
import lensmaterials as lm
import kabamland2 as k2
import numpy as np

from Geant4.hepunit import *

##### Scintillation parameters #####
Scnt_PP = np.array([ 6.6*eV, 6.7*eV, 6.8*eV, 6.9*eV, 7.0*eV, 7.1*eV, 7.2*eV, 7.3*eV, 7.4*eV ])

Scnt_FAST = np.array([ 0.000134, 0.004432, 0.053991, 0.241971, 0.398942, 0.000134, 0.004432, 0.053991, 0.241971 ])
Scnt_SLOW = np.array([ 0.000010, 0.000020, 0.000030, 0.004000, 0.008000, 0.005000, 0.020000, 0.001000, 0.000010 ])


def create_gamma_event(location, energy, amount, config, eventname, datadir=""):
    # simulates a number of single gamma photon events equal to amount
    # at position given by location for a given configuration.
    # Gamma energy is in MeV.
    kabamland = Detector(lm.ls)
    k2.build_kabamland(kabamland, config)
    # kabamland.add_solid(Solid(make.box(0.1,0.1,0.1,center=location), glass, lm.ls, color=0x0000ff)) # Adds a small blue cube at the event location, for viewing
    kabamland.flatten()
    # kabamland.bvh = load_bvh(kabamland)
    # view(kabamland)
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

if __name__ == '__main__':
	create_gamma_event((0,0,0), 1.0, 1, 'cfJiani3_4', 'event-gamma-test-(0-0-0)-1.root', datadir='')
