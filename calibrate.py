import os
import argparse
import deepdish as dd
import h5py
import numpy as np  # Just for float32 (currently)
import psutil

import paths
import detectorconfig
from DetectorResponse import DetectorResponse
from DetectorResponsePDF import DetectorResponsePDF
from DetectorResponseGaussAngle import DetectorResponseGaussAngle
import lensmaterials as lm
from logger_lfd import logger
import kabamland2 as kb   # Move uniform photons out of kabamland
import utilities

from chroma.detector import G4DetectorParameters

# TODO: Move this method and uniform_photons to utilities?
def full_detector_simulation(amount, configname, simname, datadir=""):
    # simulates 1000*amount photons uniformly spread throughout a sphere whose radius is the inscribed radius of the icosahedron.
    # Note that viewing may crash if there are too many lenses. (try using configview)
    from chroma.sim import Simulation     # Require CUDA, so only import when necessary
    from ShortIO.root_short import ShortRootWriter

    logger.info('Starting to load/build: %s' % configname)
    g4_detector_parameters=G4DetectorParameters(orb_radius=7., world_material='G4_Galactic')
    kabamland = utilities.load_or_build_detector(configname, lm.create_scintillation_material(), g4_detector_parameters=g4_detector_parameters)
    logger.info('Detector was loaded/built')

    file_name_base = datadir + simname
    f = ShortRootWriter(datadir + simname+'.root')
    # SHERLOCK: Have to set the seed because Sherlock compute machines blow up if we use chroma's algorithm!!!!
    # sim = Simulation(kabamland, geant4_processes=0, seed=65432)  # For now, does not take advantage of multiple cores  # TODO: use sim_setup()?
    sim = Simulation(kabamland, geant4_processes=0)  # For now, does not take advantage of multiple cores  # TODO: use sim_setup()?
    with h5py.File(file_name_base + '.h5','w') as h5_file:
        # Total photons will be LOOP_COUNT * EVENT_COUNT * amount
        LOOP_COUNT = 100
        EVENT_COUNT = 10

        # Setup to write the file incrementally
        # Can't use deepdish as it seems to require a single write
        start_pos = h5_file.create_dataset('photons_start', shape=(LOOP_COUNT*EVENT_COUNT, amount, 3), dtype=np.float32, chunks=True)
        end_pos = h5_file.create_dataset('photons_stop', shape=(LOOP_COUNT*EVENT_COUNT, amount, 3), dtype=np.float32, chunks=True)
        photon_flags = h5_file.create_dataset('photon_flags', shape=(LOOP_COUNT*EVENT_COUNT, amount,), dtype=np.uint32, chunks=True)

        cl = detectorconfig.DetectorConfigurationList()
        config = cl.get_configuration(configname)
        # Store the UUID and name to enable matching of the configuration, calibration, and simulation files
        h5_file.attrs['config_name'] = config.config_name
        h5_file.attrs['config_UUID'] = str(config.uuid)

        process = psutil.Process(os.getpid())
        logger.info('Memory size: %d MB' % (process.memory_info().rss // 1000000))
        for j in range(LOOP_COUNT):
            logger.info('%d of %d event sets' % (j, LOOP_COUNT))
            ev_index = 0
            sim_events = [kb.uniform_photons(config.detector_radius, amount) for i in range(EVENT_COUNT)]
            for ev in sim.simulate(sim_events, keep_photons_beg = True, keep_photons_end = True, run_daq=False, max_steps=100):
                start_pos[(j* EVENT_COUNT) + ev_index] = ev.photons_beg.pos
                end_pos[(j* EVENT_COUNT) + ev_index] = ev.photons_end.pos
                photon_flags[(j* EVENT_COUNT) + ev_index] = ev.photons_end.flags
                ev_index += 1

                # For now, write both hdf5 and ROOT simulation files
                f.write_event(ev)
            logger.info('Memory size: %d MB' % (process.memory_info().rss // 1000000))

    f.close()
    # Centralize this sort of stuff

    #h5_dict = {'config': config, 'photons_start': photons_start_pos, 'photons_stop': photons_stop_pos, 'photon_flags': photon_flags}
    # Can probably move back to a straight hdf5 file?
    #dd.io.save(file_name_base +'.h5', h5_dict)


# From detectoranalysis - remove it from there
# This should really be called "calibrate"
def create_detres_aka_calibrate(config, photons_file, detresname, detxbins=10, detybins=10, detzbins=10, method="PDF", nevents=-1, datadir="", fast_calibration=False):
    # saves a detector response list of pdfs- 1 for each pixel- given a simulation file of photons emitted isotropically throughout the detector.
    logger.info('Calibrating with: ' + datadir + photons_file)
    if method == "PDF":
        dr = DetectorResponsePDF(config, detxbins, detybins, detzbins)       # Do we need to continue to carry this?
    elif method == "GaussAngle":
        dr = DetectorResponseGaussAngle(config, detxbins, detybins, detzbins)
    else:
        logger.warning('Warning: using generic DetectorResponse base class.')
        dr = DetectorResponse(config)

    dr.calibrate(datadir + photons_file, datadir, nevents, fast_calibration=fast_calibration)
    logger.info("=== Detector analysis calibration complete.  Writing calibration file")
    # dr.calibrate_old(datadir + simname, nevents)
    # print dr.means[68260]
    # print dr.sigmas[68260]

    dr.write_to_ROOT(datadir + detresname + '.root')

    # Config dict is just for human readability (currently)
    detector_data = {'config': dr.config, 'config_dict': vars(dr.config), 'means': dr.means, 'sigmas': dr.sigmas}
    dd.io.save(datadir + detresname +'.h5', detector_data)

# From detectoranalysis - TODO: remove it from there
# saves a detector response list of pdfs-1 for each pixel-given a simulation file of photons emitted isotropically throughout the detector.
def calibrate(config, photons_file, detresname, detxbins=10, detybins=10, detzbins=10, method="PDF", nevents=-1, datadir="", fast_calibration=False):
    logger.info('Calibrating with: ' + datadir + photons_file)
    if method == "PDF":
        dr = DetectorResponsePDF(config, detxbins, detybins, detzbins)       # Do we need to continue to carry this?
    elif method == "GaussAngle":
        dr = DetectorResponseGaussAngle(config, detxbins, detybins, detzbins)
    else:
        logger.warning('Warning: using generic DetectorResponse base class.')
        dr = DetectorResponse(config)
    dr.calibrate(datadir + photons_file, datadir, nevents, fast_calibration=fast_calibration)
    logger.info("=== Detector analysis calibration complete.  Writing calibration file")

    # For now, write both hdf5 and ROOT files
    dr.write_to_ROOT(datadir + detresname + '.root')

    # Config dict is just included for human readability (currently)
    detector_data = {'config': dr.config, 'config_dict': vars(dr.config), 'means': dr.means, 'sigmas': dr.sigmas}
    dd.io.save(datadir + detresname +'.h5', detector_data)


def simulate_and_calibrate(cfg):
    if os.path.isfile(paths.get_calibration_file_name(cfg)):
        logger.info('Found calibration file: %s' % paths.get_calibration_file_name(cfg))
    else:
        logger.info('Failed to find calibration file: ' + paths.get_calibration_file_name(cfg))
        logger.info('==== Step 1: Setting up the detector ====')
        photons_file_base = 'sim-'+cfg+'_100million'
        photons_file_full_path_base = paths.detector_calibration_path + photons_file_base
        if not (os.path.exists(photons_file_full_path_base+'.root') or os.path.exists(photons_file_full_path_base+'.h5')):
            logger.info('==== Building detector and simulating photons: %s  ====' % photons_file_base)
            full_detector_simulation(100000, cfg, photons_file_base, datadir=paths.detector_calibration_path)
            simulation_file = photons_file_base + '.h5'
        elif os.path.exists(photons_file_full_path_base+'.h5'):  # TODO: The double if's and constantly adding extensions needs to be reworked
            simulation_file = photons_file_base+ '.h5'
        else:     # Fall back to root
            simulation_file = photons_file_base+ '.root'
        logger.info('==== Found/built photons file: %s ====' % simulation_file)
        logger.info("==== Step 2: Calibrating  ====")
        calibrate(args.cfg,
                simulation_file,
                paths.get_calibration_file_name_base_without_path(cfg),
                method="GaussAngle",
                nevents=10000,
                datadir=paths.detector_calibration_path,
                fast_calibration=True)
        #os.remove(photons_file)  # Would need to remove both
        logger.info("==== Calibration complete ====")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', help='configuration')
    args = parser.parse_args()
    cfg = args.cfg
    simulate_and_calibrate(cfg)

