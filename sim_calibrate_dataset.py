import os
import argparse

import kabamland2 as kb
import detectoranalysis as da
import paths
import g4_sim
import detectorconfig
import lensmaterials
import driver_utils

from logger_lfd import logger

def calibrate_and_simulate(cfg, particle, dist_range, energy ):
    if not os.path.isfile(paths.get_calibration_file_name(cfg)):   # This is not a great structure as other configuration data may change in addition to the detector config
            logger.info('Failed to find: ' + paths.get_calibration_file_name(cfg))
            # We should really date stamp the directory containing the output and configuration files
            logger.info('==== Setting up the detector ====')
            photons_file = 'sim-'+cfg+'_100million.root'
            if not os.path.exists(paths.detector_calibration_path + photons_file):
                    logger.info('==== Building detector and simulating photons ====')
                    kb.full_detector_simulation(100000, cfg, photons_file, datadir=paths.detector_calibration_path)
            logger.info("==== Calibrating  ====")
            da.create_detres(args.cfg,
                             photons_file,
                             paths.get_calibration_file_name_without_path(cfg),
                             method="GaussAngle",
                             nevents=1000,
                             datadir=paths.detector_calibration_path)
            #os.remove(photons_file)
            logger.info("==== Calibration complete ====")

    if True:
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

            # Write both files for now to support Jacopo's test setup
            driver_utils.save_config_file(cfg, 'conf.pkl', detectorconfig.configdict(cfg))  # cfg].__dict__)

    logger.info('==== Simulation part ====')
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
            calibrate_and_simulate(cfg, particle, dist_range, energy)

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

