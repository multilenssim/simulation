import os
import argparse

import kabamland2 as kb
import detectoranalysis as da
import paths
import detectorconfig
import lensmaterials as lm
from ShortIO.root_short import ShortRootWriter
from logger_lfd import logger

from chroma.detector import G4DetectorParameters
from chroma.sim import Simulation

# TOTO: Move this ,ethod and uniform_photons to driver_utils?
def full_detector_simulation(amount, configname, simname, datadir=""):
    # simulates 1000*amount photons uniformly spread throughout a sphere whose radius is the inscribed radius of the icosahedron.
    # Note that viewing may crash if there are too many lenses. (try using configview)

    config = detectorconfig.configdict(configname)
    logger.info('Starting to load/build: %s' % configname)
    g4_detector_parameters=G4DetectorParameters(orb_radius=7., world_material='G4_Galactic')
    kabamland = kb.load_or_build_detector(configname, lm.create_scintillation_material(), g4_detector_parameters=g4_detector_parameters)
    logger.info('Detector was loaded/built')

    f = ShortRootWriter(datadir + simname)
    sim = Simulation(kabamland, geant4_processes=0)  # For now, does not take advantage of multiple cores
    for j in range(100):
        logger.info('%d of 100 event sets' % j)
        sim_events = [kb.uniform_photons(config.edge_length, amount) for i in range(10)]
        for ev in sim.simulate(sim_events, keep_photons_beg = True, keep_photons_end = True, run_daq=False, max_steps=100):
            f.write_event(ev)
    f.close()


def calibrate(cfg):
    if os.path.isfile(paths.get_calibration_file_name(cfg)):
        print('Found calibration file: %s' % paths.get_calibration_file_name(cfg))
    else:
        logger.info('Failed to find calibration file: ' + paths.get_calibration_file_name(cfg))
        logger.info('==== Step 1: Setting up the detector ====')
        photons_file = 'sim-'+cfg+'_100million.root'
        if not os.path.exists(paths.detector_calibration_path + photons_file):
            logger.info('==== Building detector and simulating photons ====')
            full_detector_simulation(100000, cfg, photons_file, datadir=paths.detector_calibration_path)
        else:
            logger.info('==== Found photons file ====')            
        logger.info("==== Step 2: Calibrating  ====")
        da.create_detres(args.cfg,
                         photons_file,
                         paths.get_calibration_file_name_without_path(cfg),
                         method="GaussAngle",
                         nevents=10000,
                         datadir=paths.detector_calibration_path,
                         fast_calibration=True)
        #os.remove(photons_file)
        logger.info("==== Calibration complete ====")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', help='configuration')
    args = parser.parse_args()
    cfg = args.cfg
    calibrate(cfg)

