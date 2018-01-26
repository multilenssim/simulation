import time
import os
import argparse

import Geant4

import paths
from EventAnalyzer import EventAnalyzer
from DetectorResponseGaussAngle import DetectorResponseGaussAngle
from drivers import driver_utils

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

def generate_events(sample_count, config_name, particle, energy, i_r, o_r):
    # File pathing stuff should not be in here
    seed_loc = 'r%i-%i' % (i_r, o_r)
    data_file_dir = paths.get_data_file_path_no_raw(config_name)
    if not os.path.exists(data_file_dir):
        os.makedirs(data_file_dir)
    fname_base = data_file_dir+seed_loc+'_'+str(energy)+'_'+particle+'_'+'sim'
    fname = fname_base+'.h5'


    print('Configuration loaded: ' + config_name)
    print('Energy: ' + str(energy))
    print("G4 state: ", Geant4.gStateManager.GetCurrentState())
    print("Random engine: ", Geant4.HepRandom.getTheEngine())
    print("Random seed: ", Geant4.HepRandom.getTheSeed())

    driver_utils.fire_g4_particles(sample_count, config_name, particle, energy, i_r, o_r, fname, di_file_base=fname_base)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', help='detector configuration')
    parser.add_argument('particle', help='particle to simulate')
    parser.add_argument('s_d', help='seed location')
    args = parser.parse_args()

    #for particle in ['neutron']: # ['e-']:  # ,'gamma']:
    #for dist_range in ['01']:  #,'34']:
    sample = 5
    #energy = 50.
    start_time = time.time()
    print('CUDA initialized')
    for energy in [2,10,50]:
        generate_events(sample, args.cfg, args.particle, energy, int(args.s_d[0]), int(args.s_d[1]))
