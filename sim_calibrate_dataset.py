import os
import pickle
import argparse

import kabamland2 as kb
import detectoranalysis as da
import paths
import g4_sim
import detectorconfig
import lensmaterials

from logger_lfd import logger


parser = argparse.ArgumentParser()
parser.add_argument('cfg', help='configuration')
# parser.add_argument('s_d', help='Seed radius range in meters (e.g. "34"')
args = parser.parse_args()
cfg = args.cfg
# s_d = args.s_d  # Seed radius range in meters

def save_config_file(cfg, file_name, dict):
        config_path = paths.get_data_file_path(cfg)
        if not os.path.exists(config_path):
                os.makedirs(config_path)
        with open(config_path+file_name, 'w') as outf:
                pickle.dump(dict, outf)

photons_file = 'sim-'+cfg+'_100million.root'

if not os.path.isfile(paths.get_calibration_file_name(cfg)):   # This is not a great structure as other configuration data may change in addition to the detector config
        logger.info('Failed to find: ' + paths.get_calibration_file_name(cfg))
        # We should really date stamp the directory containing the output and configuration files
	logger.info('==== Setting up the detector ====')
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
        all_config_info = {'configuration': detectorconfig.configdict[cfg].__dict__}
        all_config_info['scintillator'] = lensmaterials.create_scintillation_material().__dict__
        all_config_info['lens_material'] = lensmaterials.lensmat.__dict__
        all_config_info['G4_config'] = 'placeholder'
        all_config_info['particle_config'] = 'placeholder'  # Needs to include energies
        all_config_info['quantum_efficiency'] = 'placeholder'
        save_config_file(cfg, 'full_config.pickle', all_config_info)

        # Write both files for now to support Jacopo's test setup
        save_config_file(cfg, 'conf.pkl', detectorconfig.configdict[cfg].__dict__)

logger.info('==== Simulation part ====')
for dist_range in ['01','34']:
        g4_sim.run_simulation(cfg, 'e-', dist_range)
        g4_sim.run_simulation(cfg, 'gamma', dist_range)

#os.system('python g4_sim.py e- %s %s'%(s_d,cfg))
#os.system('python g4_sim.py gamma %s %s'%(s_d,cfg))
