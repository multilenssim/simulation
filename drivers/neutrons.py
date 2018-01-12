import pickle
#import matplotlib.pyplot as plt
import numpy as np
import h5py
import time
import os
import argparse

import paths
import nog4_sim
from DetectorResponseGaussAngle import DetectorResponseGaussAngle
from EventAnalyzer import EventAnalyzer

from chroma.generator import vertex

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
    sim, analyzer = nog4_sim.sim_setup(config, detres)  # KW: where did this line come from?  It seems to do nothing
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



def plot_vertices(track_tree, title, with_electrons=True, file_name='vertex_plot.pickle'):
    particles = {}
    energies = {}
    for key, value in track_tree.iteritems():
        if 'particle' in value:
            particle = value['particle']
            if particle not in particles:
                particles[particle] = []
                energies[particle] = []
            particles[particle].append(value['position'])       # Not sure if this will work??  Changed the Chroma track_tree API
            energies[particle].append(100.*value['energy'])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax = fig.gca(projection='3d')

    for key, value in particles.iteritems():
        if with_electrons or key != 'e-':
            the_array = np.array(value)
            #ax.plot(the_array[:,0], the_array[:,1], the_array[:,2], '.', markersize=5.0)
            ax.scatter(the_array[:,0], the_array[:,1], the_array[:,2], marker='o', s=energies[particle], label=key) #), markersize=5.0)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    #if args.hdf5 is None:
    #    ax.plot(vtx[:, 0], vtx[:, 1], vtx[:, 2], '.')
    plt.legend(loc=2)   # See https://pythonspot.com/3d-scatterplot/

    # See: http://fredborg-braedstrup.dk/blog/2014/10/10/saving-mpl-figures-using-pickle
    pickle.dump(fig, file(file_name, 'wb'))      # Shouldn't this be 'wb'?
    plt.show()


def write_h5_reverse_track_file_event(h5file, vert, tracks, first):
        if first:
            en_depo = h5file.create_dataset('en_depo', maxshape=(None, 3), data=vert, chunks=True)
            coord = h5file.create_dataset('coord', maxshape=(2, None, 3),
                                          data=[tracks.hit_pos.T, tracks.means.T], chunks=True)
            uncert = h5file.create_dataset('sigma', maxshape=(None,), data=tracks.sigmas, chunks=True)
            h5file.create_dataset('r_lens', data=tracks.lens_rad)
            h5file.create_dataset('idx_tr', data=uncert.shape[0])
            h5file.create_dataset('idx_depo', data=en_depo.shape[0])
        else:
            en_depo = h5file.get('en_depo')      # This may be super inefficient??
            en_depo.resize(en_depo.shape[0] + vert.shape[0], axis=0)
            en_depo[-vert.shape[0]:, :] = vert
            coord = h5file.get('coord')
            coord.resize(coord.shape[1] + tracks.means.shape[1], axis=1)
            coord[:, -tracks.means.shape[1]:, :] = [tracks.hit_pos.T, tracks.means.T]
            uncert = h5file.get('uncert')
            uncert.resize(uncert.shape[0] + tracks.sigmas.shape[0], axis=0)
            uncert[-tracks.sigmas.shape[0]:] = tracks.sigmas

            # Untested - and is there a better way?  This looks too complicated
            idx_tr = h5file.get('idx_tr')
            idx_tr_size = idx_tr.shape
            idx_tr.resize(idx_tr_size + 1)
            idx_tr[idx_tr_size] = uncert.shape[0]

            idx_depo = h5file.get('idx_depo')
            idx_depo_size = idx_depo.shape
            idx_depo.resize(idx_depo_size + 1)
            idx_depo[idx_depo_size] = en_depo.shape[0]

def generate_events(sample, cfg, particle, energy, i_r, o_r):
    # File pathing stuff should not be in here
    seed_loc = 'r%i-%i' % (i_r, o_r)
    data_file_dir = paths.get_data_file_path(cfg)
    if not os.path.exists(data_file_dir):
        os.makedirs(data_file_dir)
    fname = data_file_dir + seed_loc + '_' + str(energy) + particle + '_' + 'sim.h5'

    sim, analyzer = nog4_sim.sim_setup(cfg, paths.get_calibration_file_name(cfg), useGeant4=True)
    print('Configuration loaded: ' + cfg)
    print('Energy: ' + str(energy))
    location = nog4_sim.sph_scatter(sample, i_r * 1000, o_r * 1000)
    with h5py.File(fname, 'w') as f:
        first = True
        print('Running locations: ' + str(len(location)))
        for lg in location:
            start = time.time()
            gun = vertex.particle_gun([particle], vertex.constant(lg), vertex.isotropic(),
                                  vertex.flat(float(energy) * 0.999, float(energy) * 1.001))
            for ev in sim.simulate(gun, keep_photons_beg=True, keep_photons_end=True, run_daq=False, max_steps=100):
                vert = ev.photons_beg.pos
                tracks = analyzer.generate_tracks(ev, qe=(1. / 3.))
                write_h5_reverse_track_file_event(f, vert, tracks, first)
                first = False

            print ('Time: ' + str(time.time() - start) + '\tPhotons detected: ' + str(tracks.sigmas.shape[0]))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', help='detector configuration')
    parser.add_argument('particle', help='particle to simulate')
    parser.add_argument('s_d', help='seed location')
    args = parser.parse_args()

    #for particle in ['neutron']: # ['e-']:  # ,'gamma']:
    #for dist_range in ['01']:  #,'34']:
    sample = 500
    energy = 20.
    start_time = time.time()
    print('CUDA initialized')
    generate_events(sample, args.cfg, args.particle, energy, int(args.s_d[0]), int(args.s_d[1]))
