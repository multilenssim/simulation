import os
import pickle
import argparse

import kabamland2 as kb
import detectoranalysis as da
import lensmaterials
import detectorconfig

from chroma.log import logger as chroma_logger
from logger_lfd import logger

import paths

parser = argparse.ArgumentParser()
parser.add_argument('cfg', help='configuration')
args = parser.parse_args()
cfg = args.cfg
s_d='01'  # Seed radius range in meters

photons_file = 'sim-'+cfg+'_100million.root'

if not os.path.isfile(paths.get_calibration_file_name(cfg)):   # This is not a great structure as other configuration data may change in addition to the detector config
        logger.info('Failed to find: ' + paths.get_calibration_file_name(cfg))
        # We should really date stamp the directory containing the output and configuration files
	logger.info('==== Setting up the detector ====')
        if not os.path.exists(paths.detector_calibration_path + photons_file):
	        logger.info('==== Building detector and simulating photons ====')
                kb.full_detector_simulation(100000, cfg, photons_file, datadir=paths.detector_calibration_path)
        logger.info("==== Calibrating  ====")
        da.create_detres(args.cfg, photons_file,
                         paths.get_calibration_file_name_without_path(cfg),
                         method="GaussAngle",
                         nevents=1000,
                         datadir=paths.detector_calibration_path)
        #os.remove(photons_file)
        logger.info("==== Calibration complete ====")

if True:
        config_path = paths.get_data_file_path(cfg)
        if not os.path.exists(config_path):
                os.makedirs(config_path)
        all_config_info = {'configuration': detectorconfig.configdict[cfg].__dict__}
        all_config_info['scintillator'] = lensmaterials.create_scintillation_material().__dict__
        all_config_info['lens_material'] = lensmaterials.lensmat.__dict__
        all_config_info['G4_config'] = 'placeholder'
        all_config_info['particle_config'] = 'placeholder'
        with open(config_path+'full_config.pickle', 'w') as outf:
                pickle.dump(all_config_info, outf)

        # Write both files for now to support Jacopo's test setup
        with open(config_path+'conf.pkl', 'w') as outf:
                pickle.dump(detectorconfig.configdict[cfg].__dict__, outf)

logger.info('==== Simulation part ====')
os.system('python g4_sim.py e- %s %s'%(s_d,cfg))
os.system('python g4_sim.py gamma %s %s'%(s_d,cfg))
