import time
import os
import argparse

import Geant4

import paths
from EventAnalyzer import EventAnalyzer
from DetectorResponseGaussAngle import DetectorResponseGaussAngle
import driver_utils


######## This is old code - clean it out....

# This is the call from efficiency.py:
#	eff_test(detfile,
# 		detres=paths.get_calibration_file_name(detfile),
# 		detbins=10,
# 		sig_pos=0.01,
# 		n_ph_sim=energy,
# 		repetition=repetition,
# 		max_rad=6600,
# 		n_pos=n_pos,
# 		loc1=(0,0,0),
# 		sig_cone=0.01,
# 		lens_dia=None,
# 		n_ph=0,
# 		min_tracks=0.1,
# 		chiC=1.5,
# 		temps=[256, 0.25],
# 		tol=0.1,
# 		debug=False)

def simulate_and_compute_AVF(config, detres=None):
    sim, analyzer = driver_utils.sim_setup(config, detres)  # KW: where did this line come from?  It seems to do nothing
    detbins = 10

    if detres is None:
        det_res = DetectorResponseGaussAngle(config, detbins, detbins, detbins)
    else:
        det_res = DetectorResponseGaussAngle(config, detbins, detbins, detbins, infile=detres)

    amount = 5333
    sig_pos = 0.01
    rad = 1.0  # Location of event - will be DEPRECATED

    analyzer = EventAnalyzer(det_res)
    events, points = create_single_source_events(rad, sig_pos, amount, repetition)

    sig_cone = 0.01
    lens_dia = None
    n_ph = 0
    min_tracks = 0.1
    chiC = 1.5
    temps = [256, 0.25]
    tol = 0.1
    debug = True

    for ind, ev in enumerate(sim.simulate(events, keep_photons_beg=True, keep_photons_end=True, run_daq=False, max_steps=100)):
        # Do AVF event reconstruction
        vtcs = analyzer.analyze_one_event_AVF(ev, sig_cone, n_ph, min_tracks, chiC, temps, tol, debug, lens_dia)

def create_double_fixed_source_events(loc1, loc2, amount1, amount2):
    import kabamland2 as kbl

    # produces a list of Photons objects, each with two different photon sources at fixed locations
    events = []
    events.append(kbl.constant_photons(loc1, int(amount1)) + kbl.constant_photons(loc2, int(amount2)))
    return events

def run_simulation_double_fixed_source(sample, cfg, loc1, loc2, amount, qe=None):
    import h5py
    import numpy as np

    # File pathing stuff should not be in here
    data_file_dir = paths.get_data_file_path_no_raw(cfg)
    if not os.path.exists(data_file_dir):
        os.makedirs(data_file_dir)
    dist = np.linalg.norm(loc2 - loc1)
    fname_base = data_file_dir + 'd-site-'+str(int(dist/10))+'cm-'+str(qe)
    fname = fname_base + '.h5'

    sim, analyzer = driver_utils.sim_setup(cfg, paths.get_calibration_file_name(cfg), useGeant4=True, geant4_processes=1)

    print('Configuration loaded: ' + cfg)
    print('Photon count: ' + str(amount))
    print("G4 state: ", Geant4.gStateManager.GetCurrentState())
    print("Random engine: ", Geant4.HepRandom.getTheEngine())
    print("Random seed: ", Geant4.HepRandom.getTheSeed())

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

                # vertices = utilities.AVF_analyze_event(analyzer, ev)
                # utilities.plot_vertices(ev.photons_beg.track_tree, 'AVF plot', reconstructed_vertices=vertices)

                gun_specs = driver_utils.build_gun_specs(None, loc1, None, amount)      # TODO: Need loc2?? & using amount of photons as energy
                di_file = driver_utils.DIEventFile(cfg, gun_specs, ev.photons_beg.track_tree, tracks, photons=ev.photons_beg, full_event=ev)
                di_file.write(fname_base + '_' + str(i) + '.h5')

                first = False
                i += 1

            print ('Time: ' + str(time.time() - start) + '\tPhotons detected: ' + str(tracks.sigmas.shape[0]))


def generate_events_old_but_working(sample, cfg, particle, energy, i_r, o_r, qe=None):
    # File pathing stuff should not be in here
    seed_loc = 'r%i-%i' % (i_r, o_r)
    data_file_dir = paths.get_data_file_path_no_raw(cfg)
    if not os.path.exists(data_file_dir):
        os.makedirs(data_file_dir)
    fname_base = data_file_dir+seed_loc+'_'+str(energy)+'_'+particle+'_'+str(qe)+'_sim'
    fname = fname_base+'.h5'

    sim, analyzer = driver_utils.sim_setup(cfg, paths.get_calibration_file_name(cfg), useGeant4=True, geant4_processes=1)

    print('Configuration loaded: ' + cfg)
    print('Energy: ' + str(energy))
    print("G4 state: ", Geant4.gStateManager.GetCurrentState())
    print("Random engine: ", Geant4.HepRandom.getTheEngine())
    print("Random seed: ", Geant4.HepRandom.getTheSeed())

    location = driver_utils.sph_scatter(sample, i_r * 1000, o_r * 1000)
    #print('Loc: ' + str(location))
    location = [(0,0,0)]
    i = 0

    import h5py
    from chroma.generator import vertex
    import numpy as np

    with h5py.File(fname, 'w') as f:
        first = True
        print('Running locations: ' + str(len(location)))
        for i in range(sample): # lg in location:
        #for lg in location:
            lg = location[0]
            start = time.time()
            gun = vertex.particle_gun([particle], vertex.constant(lg), vertex.constant(np.array([1,0,0])),   #isotropic(),
                                  vertex.constant(energy))  #flat(float(energy) * 0.999, float(energy) * 1.001))
            events = sim.simulate(gun, keep_photons_beg=True, keep_photons_end=True, run_daq=False, max_steps=100)
            for ev in events:
                vert = ev.photons_beg.pos
                tracks = analyzer.generate_tracks(ev, qe=qe)
                driver_utils.write_h5_reverse_track_file_event(f, vert, tracks, first)

                #vertices = utilities.AVF_analyze_event(analyzer, ev)
                #utilities.plot_vertices(ev.photons_beg.track_tree, 'AVF plot', reconstructed_vertices=vertices)

                gun_specs = driver_utils.build_gun_specs(particle, lg, None, energy)
                di_file = driver_utils.DIEventFile(cfg, gun_specs, ev.photons_beg.track_tree, tracks, photons=ev.photons_beg, full_event=ev)
                di_file.write(fname_base+'_'+str(i)+'.h5')

                first = False
                i += 1

            print ('Time: ' + str(time.time() - start) + '\tPhotons detected: ' + str(tracks.sigmas.shape[0]))

def generate_events(sample_count, config_name, particle, energy, i_r, o_r, qe=None):
    # File pathing stuff should not be in here
    seed_loc = 'r%i-%i' % (i_r, o_r)
    data_file_dir = paths.get_data_file_path_no_raw(config_name)
    if not os.path.exists(data_file_dir):
        os.makedirs(data_file_dir)
    fname_base = data_file_dir+seed_loc+'_'+str(energy)+'_'+particle+'_'+str(qe)+'_sim'
    fname = fname_base+'.h5'

    print('Configuration loaded: ' + config_name)
    print('Energy: ' + str(energy))
    print("G4 state: ", Geant4.gStateManager.GetCurrentState())
    print("Random engine: ", Geant4.HepRandom.getTheEngine())
    print("Random seed: ", Geant4.HepRandom.getTheSeed())

    driver_utils.fire_g4_particles(sample_count, config_name, particle, energy, i_r, o_r, fname, di_file_base=fname_base, qe=qe)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', help='detector configuration')
    parser.add_argument('particle', help='particle to simulate')
    parser.add_argument('s_d', help='seed location')
    args = parser.parse_args()

    #for particle in ['neutron']: # ['e-']:  # ,'gamma']:
    #for dist_range in ['01']:  #,'34']:
    _sample = 3
    #energy = 50.
    #for energy in [2,10,50]:
    #    generate_events_old_but_working(_sample, args.cfg, args.particle, energy, int(args.s_d[0]), int(args.s_d[1]))

    run_simulation_double_fixed_source(_sample, args.cfg, (0,0,0), (2,0,0), 16000, qe=None)
