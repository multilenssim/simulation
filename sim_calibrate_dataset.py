import os
import argparse

import kabamland2 as kb
import detectoranalysis as da
import paths
import g4_sim
import detectorconfig
import lensmaterials
import driver_utils
import lensmaterials as lm
from ShortIO.root_short import ShortRootWriter

from chroma.detector import G4DetectorParameters
from chroma.sim import Simulation

from logger_lfd import logger

# Move/Use driver_utils for this?
# And for uniform_photoms?
def full_detector_simulation(amount, configname, simname, datadir=""):
    #simulates 1000*amount photons uniformly spread throughout a sphere whose radius is the inscribed radius of the icosahedron. Note that viewing may crash if there are too many lenses. (try using configview)

    config = detectorconfig.configdict(configname)
    print('Starting to build')
    g4_detector_parameters=G4DetectorParameters(orb_radius=7., world_material='G4_Galactic')
    kabamland = kb.load_or_build_detector(configname, lm.create_scintillation_material(), g4_detector_parameters=g4_detector_parameters)
    print('Detector was built')

    f = ShortRootWriter(datadir + simname)
    sim = Simulation(kabamland,geant4_processes=0)
    for j in range(100):
        print j
        sim_events = [kb.uniform_photons(config.edge_length, amount) for i in range(10)]
        for ev in sim.simulate(sim_events, keep_photons_beg = True, keep_photons_end = True, run_daq=False, max_steps=100):
            f.write_event(ev)
    f.close()


def calibrate(cfg, particle, dist_range, energy):
    if not os.path.isfile(paths.get_calibration_file_name(cfg)):   # This is not a great structure as other configuration data may change in addition to the detector config
        logger.info('Failed to find: ' + paths.get_calibration_file_name(cfg))
        # We should really date stamp the directory containing the output and configuration files
        logger.info('==== Step 1: Setting up the detector ====')
        photons_file = 'sim-'+cfg+'_1billion.root'
        if not os.path.exists(paths.detector_calibration_path + photons_file):
                logger.info('==== Step 1.1: Building detector and simulating photons ====')
                full_detector_simulation(1000000, cfg, photons_file, datadir=paths.detector_calibration_path)
        logger.info("==== Step 2: Calibrating  ====")
        da.create_detres(args.cfg,
                         photons_file,
                         paths.get_calibration_file_name_without_path(cfg),
                         method="GaussAngle",
                         nevents=10000,
                         datadir=paths.detector_calibration_path)
        #os.remove(photons_file)
        logger.info("==== Calibration complete ====")
    else:
        print('Found calibration file: %s' % paths.get_calibration_file_name(cfg))

    # Continue to write this to support Jacopo's test setup
    driver_utils.save_config_file(cfg, 'conf.pkl', detectorconfig.configdict(cfg))  # cfg].__dict__)

    if False:     # Need to reexamine this in light of what we are doing - is this a cal config, or is this a simulation config
        all_config_info = {'configuration': detectorconfig.configdict(cfg)}
        # all_config_info = {'configuration': detectorconfig.configdict[cfg].__dict__}  # Old design - are we suppoting anymore?
        all_config_info['scintillator'] = lensmaterials.create_scintillation_material().__dict__
        all_config_info['lens_material'] = lensmaterials.lensmat.__dict__
        all_config_info['G4_config'] = 'placeholder'
        # Note - these might override on successive runs??
        all_config_info['particle'] = particle
        all_config_info['energy'] = energy
        all_config_info['distance_range'] = dist_range
        all_config_info['quantum_efficiency'] = 'placeholder'
        driver_utils.save_config_file(cfg, 'full_config.pickle', all_config_info)

def simulate(cfg, particle, dist_range, energy):
    logger.info('==== Step 3: Simulation part ====')
    g4_sim.run_simulation(cfg, particle, dist_range, energy)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', help='configuration')
    # parser.add_argument('s_d', help='Seed radius range in meters (e.g. "34"')
    args = parser.parse_args()
    cfg = args.cfg
    # s_d = args.s_d  # Seed radius range in meters
    energy = 2.
    for dist_range in ['01', '34']:
        for particle in ['e-', 'gamma']:
            calibrate(cfg, particle, dist_range, energy)
            simulate(cfg, particle, dist_range, energy)

def jacopos_version():
    import os, itertools

    l_base = [1, 8]
    EPDR = [8]
    configs = ['cfSam1_K%i_%i' % (k[0], k[1]) for k in list(itertools.product(l_base, EPDR))]

    for cfg in configs:
        print '----------------------------------------------------------------%s----------------------------------------------------------------' % cfg
        for s_d in ['01', '34']:
            if os.path.exists(paths.get_data_file_path(cfg)):
                print 'simulation part'
                os.system('python g4_sim.py e- %s %s' % (s_d, cfg))
                os.system('python g4_sim.py gamma %s %s' % (s_d, cfg))

            else:
                print 'setting up the detector'
                os.system('python scripts_stanford.py ' + cfg + ' full_detector')
                os.system('python scripts_stanford.py ' + cfg + ' detres')
                os.system('python save_conf.py ' + cfg)
                print 'simulation part'
                os.system('python g4_sim.py e- %s %s' % (s_d, cfg))
                os.system('python g4_sim.py gamma %s %s' % (s_d, cfg))

