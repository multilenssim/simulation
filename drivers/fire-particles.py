import time
import os
import argparse
import numpy as np

import paths
from EventAnalyzer import EventAnalyzer
import driver_utils
from logger_lfd import logger

def create_double_fixed_source_events(loc1, loc2, amount1, amount2):
    import kabamland2 as kbl

    # produces a list of Photons objects, each with two different photon sources at fixed locations
    events = []
    events.append(kbl.constant_photons(loc1, int(amount1)) + kbl.constant_photons(loc2, int(amount2)))
    return events

# Pass in sim and analyzer so don't have to recreate them each time (performace optimization)
# Do we need to pass in both?
def run_simulation_double_fixed_source(sim, analyzer, sample, cfg, loc1, loc2, amount, qe=None):
    import h5py
    import numpy as np

    # File pathing stuff should not be in here
    data_file_dir = paths.get_data_file_path_no_raw(cfg)
    if not os.path.exists(data_file_dir):
        os.makedirs(data_file_dir)
    dist = np.linalg.norm(loc2 - loc1)
    fname_base = data_file_dir + 'd-site-'+str(int(dist/10))+'cm-'+str(qe)
    fname = fname_base + '.h5'

    # sim, analyzer = driver_utils.sim_setup(cfg, paths.get_calibration_file_name(cfg), useGeant4=True, geant4_processes=1)

    logger.info('Configuration loaded: %s' % cfg)
    logger.info('Photon count: %d' % amount)

    import Geant4
    logger.info('G4 state: %s' % Geant4.gStateManager.GetCurrentState())
    logger.info('G4 Random engine: %s' % Geant4.HepRandom.getTheEngine())
    logger.info('G4 Random seed: %s' % Geant4.HepRandom.getTheSeed())

    with h5py.File(fname, 'w') as f:
        first = True
        for i in range(sample):  # lg in location:
            # for lg in location:
            start = time.time()
            sim_events = create_double_fixed_source_events(loc1, loc2, amount/2, amount/2)
            for ev in sim.simulate(sim_events, keep_photons_beg=True, keep_photons_end=True, run_daq=False,max_steps=100):
                vert = ev.photons_beg.pos
                tracks = analyzer.generate_tracks(ev, qe=qe)
                driver_utils.write_h5_reverse_track_file_event(f, vert, tracks, first)

                vertices = driver_utils.AVF_analyze_event(analyzer, ev, debug=True)
                #utilities.plot_vertices(ev.photons_beg.track_tree, 'AVF plot', reconstructed_vertices=vertices)

                gun_specs = driver_utils.build_gun_specs(None, loc1, None, amount)      # TODO: Need loc2?? & using amount of photons as energy
                di_file = driver_utils.DIEventFile(cfg, gun_specs, ev.photons_beg.track_tree, tracks, photons=ev.photons_beg, full_event=ev)
                di_file.write(fname_base + '_' + str(i) + '.h5')

                first = False
                i += 1

            logger.info('Time: ' + str(time.time() - start) + '\tPhotons detected: ' + str(tracks.sigmas.shape[0]))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='detector configuration', nargs='?', default='cfSam1_K200_10')
    parser.add_argument('particle', help='particle to simulate')
    parser.add_argument('s_d', help='seed location')
    _args = parser.parse_args()

    inner_radius = int(_args.s_d[0])
    outer_radius = int(_args.s_d[1])
    seed_loc = 'r%i-%i' % (inner_radius, outer_radius)

    particle = _args.particle
    config = _args.config

    data_file_dir = paths.get_data_file_path_no_raw(config)
    if not os.path.exists(data_file_dir):
        os.makedirs(data_file_dir)

    # Debugging only - shouldn't need Geant4 if we are just firing photons
    import Geant4
    logger.info('G4 state: %s' % Geant4.gStateManager.GetCurrentState())
    logger.info('G4 Random engine: %s' % Geant4.HepRandom.getTheEngine())
    logger.info('G4 Random seed: %s' % Geant4.HepRandom.getTheSeed())

    #for particle in ['neutron']: # ['e-']:  # ,'gamma']:
    #for dist_range in ['01']:  #,'34']:
    _sample_count = 4

    if not particle == 'photon':
        # Pass sim, analyzer in to avoid reloading the detector and improve improve performance
        #sim, analyzer = driver_utils.sim_setup(config, paths.get_calibration_file_name(config), useGeant4=False, geant4_processes=1)
        for energy in [2]: # ,10,50]:
            qe=None
            fname_base = data_file_dir+seed_loc+'_'+str(energy)+'_'+particle+'_'+str(qe)+'_sim'
            fname = fname_base+'.h5'
            #generate_events_old_but_working(_sample, args.config, args.particle, energy, int(args.s_d[0]), int(args.s_d[1]))
            #generate_events(_sample, config, particle, energy, int(args.s_d[0]), int(args.s_d[1]))
            driver_utils.fire_g4_particles(_sample_count, config, particle, energy,
                                           inner_radius, outer_radius, fname, di_file_base=fname_base, qe=qe,
                                           location=np.array([0.,0.,0.]), momentum=np.array([1.,0.,0.]))

    else:  # Photons only - need to clean this up - what optiona?  Single vs. double site?
        for dist in [2000.]: # 1., 500.,1000.,1500.,2000.]:
            sim, analyzer = driver_utils.sim_setup(config, paths.get_calibration_file_name(config), useGeant4=False)
            run_simulation_double_fixed_source(sim, analyzer, _sample_count, config, np.array([0.,0.,0.]), np.array([dist,0.,0.]), 16000, qe=None)

