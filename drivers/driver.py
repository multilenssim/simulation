import sys
import argparse
import numpy as np

from chroma.sim import Simulation
from chroma.detector import G4DetectorParameters
from chroma.generator import vertex

import lensmaterials as lm
import paths
from logger_lfd import logger
import count_processes
import driver_utils

import pycuda.driver as cuda


'''
pi = 3.1
hbarc = 4.3
nanometer = 1e-9
ls_refractive_index = 1.5
material = lm.ls

energy = list((2*pi*hbarc / (material.refractive_index[::-1,0] * nanometer)).astype(float))
foo = list([1.0][::,-1,1].astype(float))
'''

def cuda_stat():
    cuda.init()
    ndevices = cuda.Device.count()
    print('CUDA/GPU device count: ' + str(ndevices))
    # Can we assume that they are in linear order??

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

def create_gamma_event(location, energy, amount, config, eventname):
    # simulates a number of single gamma photon events equal to amount
    # at position given by location for a given configuration.
    # Gamma energy is in MeV.
    g4_detector_parameters=G4DetectorParameters(orb_radius=7., world_material='G4_Galactic')
    kabamland = driver_utils.load_or_build_detector(config, lm.create_scintillation_material(), g4_detector_parameters=g4_detector_parameters)

    sim = Simulation(kabamland, geant4_processes=1)
    logger.info('Starting gun simulation:' + paths.data_files_path + eventname)
    gun = vertex.particle_gun(['gamma'] * amount, vertex.constant(location), vertex.isotropic(),
                              vertex.flat(float(energy) * 0.99, float(energy) * 1.01))
    for ev in sim.simulate(gun, keep_photons_beg=True, keep_photons_end=True, run_daq=False, max_steps=100):
        # print 'End photons: ' + str(ev.photons_end)

        scint_count, cherenkov_count = count_processes.count_processes(ev.photons_beg, print_counts=False)

        logger.info('Photon counts (total/scintillation/cherenkov): ' + str(ev.nphotons) + ' ' +
              str(scint_count) + ' ' + str(cherenkov_count))
        if ev.nphotons != (scint_count + cherenkov_count):
            logger_lfd.logger.info('===>>> Uh oh: counts don''t add up: ' + str(scint_count) + ' ' + str(cherenkov_count))
        # print("Process types: ", pprint.pformat(type_bins))
        # print("Process subtypes: ", pprint.pformat(subtype_bins))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('configuration', help='Detector configuration')
    args = parser.parse_args()
    config = args.configuration

    cuda_stat()
    create_gamma_event((0, 0, 0), 2., 4, config, 'gamma-test-')

# From Scott's scripts_stanford.py

    # To build the calibration file
    # fileinfo='cfJiani3_10kw' # And replace all the Jiani's below
    #k2.full_detector_simulation(100000, 'cfJiani3_10kw', paths.get_calibration_file_name(fileinfo), datadir=datadir)

# For an inscribed radius of ~7400, (3000, 3000, 3000) corresponds to a radius of about 70% of the inscribed radius
#kb.create_event((3000,3000,3000), 0.1, 100000, 'cfJiani3_4', 'event-'+fileinfo+'-(3000-3000-3000)-100000.root', datadir=datadir)

#kb.create_electron_event((0,0,0), 5.0, 1, 'cfJiani3_4', 'event-electron-test-'+fileinfo+'-(0-0-0)-100000.root', datadir=datadir)

#kb.create_gamma_event((0,0,0), 1.0, 1, 'cfJiani3_4', 'event-gamma-test-'+fileinfo+'-(0-0-0)-1.root', datadir=datadir)

#kb.create_event((0, 0, 0), 0.1, 100000, 'cfSam1_1', 'event-'+fileinfo+'-(0-0-0)-100000.root', datadir=datadir)

    
#da.check_detres_sigmas('cfJiani3_4', paths.get_calibration_file_name(fileinfo), datadir=datadir)

#da.compare_sigmas('cfJiani3_4',paths.get_calibration_file_name(fileinfo),'cfJiani3_4','detresang-'+fileinfo+'_1DVariance_noreflect_100million.root',datadir=datadir)

#da.get_AVF_performance('cfJiani3_2', 'event-'+fileinfo+'-(0-0-0)-100000.root', detres=paths.get_calibration_file_name(fileinfo), detbins=20, n_repeat=5, event_pos=[(0.,0.,0.)], n_ph=[100, 1000, 10000], min_tracks=[0.1], chiC=[2.0], temps=[[256,0.25]], debug=False, datadir=datadir)

#da.reconstruct_event_AVF('cfJiani3_2', 'event-'+fileinfo+'-(0-0-0)-100000.root', detres=paths.get_calibration_file_name(fileinfo), event_pos=(0,0,0), detbins=20, chiC=2., min_tracks=0.1, n_ph=100, debug=True, datadir=datadir)
