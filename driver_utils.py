import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Required for projection='3d' below
import pickle
import numpy as np
import deepdish as dd
import argparse
import h5py
import os
import pprint

import detectorconfig  # No longer pulls in Geant4 by commenting out a LOT of imports
import lensmaterials as lm

import paths
import logger_lfd as logger

def sim_setup(config,in_file, useGeant4=False, geant4_processes=4, seed=12345, cuda_device=None):
    # Imports are here both to avoind loading Geant4 when unnecessary, and to avoid circular imports
    import kabamland2 as kbl2
    from chroma.detector import G4DetectorParameters
    from chroma.sim import Simulation
    import DetectorResponseGaussAngle
    import EventAnalyzer

    g4_detector_parameters = G4DetectorParameters(orb_radius=7., world_material='G4_Galactic') if useGeant4 else None
    kabamland = kbl2.load_or_build_detector(config, lm.create_scintillation_material(), g4_detector_parameters=g4_detector_parameters)
    sim = Simulation(kabamland, seed=seed, geant4_processes=geant4_processes if useGeant4 else 0, cuda_device=cuda_device)
    det_res = DetectorResponseGaussAngle.DetectorResponseGaussAngle(config,10,10,10,in_file)
    analyzer = EventAnalyzer.EventAnalyzer(det_res)
    return sim, analyzer

def sph_scatter(sample_count,in_shell,out_shell):
    print('sph_scatter shell radii: ' + str(in_shell) + ' ' + str(out_shell))
    loc = np.random.uniform(-out_shell,out_shell,(sample_count,3))
    while len(loc[(np.linalg.norm(loc,axis=1)>in_shell) & (np.linalg.norm(loc,axis=1)<=out_shell)]) != sample_count:
        bl_idx = np.logical_not((np.linalg.norm(loc,axis=1)>in_shell) & (np.linalg.norm(loc,axis=1)<=out_shell))
        smpl = sum(bl_idx)
        loc[bl_idx] = np.random.uniform(-out_shell,out_shell,(smpl,3))
    return loc

# Fire Geant4 particles within a spherical shell, or from a specific location
# Writes both DIEventFile (one per sample_count if file name is provided) and original HDF5 file
# 'location' is a flag as to whether to generate random locations, momentum, and energy or not
# If location is provided, those parameters will be fixed, and sample_count will be ignored
def fire_g4_particles(sample_count, config_name, particle, energy, inner_radius, outer_radius, h5_file, location= None, momentum=None, di_file_base=None):
    from chroma.generator import vertex

    sim, analyzer = sim_setup(config_name, paths.get_calibration_file_name(config_name), useGeant4=True, geant4_processes=1)

    print('Configuration loaded: ' + config_name)
    print('Energy: ' + str(energy))

    if location is None:
        loc_array = sph_scatter(sample_count, inner_radius * 1000, outer_radius * 1000)
    else:
        loc_array = [location]

    i = 0
    with h5py.File(h5_file, 'w') as f:
        first = True
        print('Running locations: ' + str(len(loc_array)))
        for i in range(sample_count):
            if location is None:  # Use location as a flag
                gun = vertex.particle_gun([particle], vertex.constant(location), vertex.isotropic(), vertex.flat(float(energy) * 0.999, float(energy) * 1.001))
            else:
                gun = vertex.particle_gun([particle], vertex.constant(location), vertex.constant(np.array(momentum)), vertex.constant(energy))

            events = sim.simulate(gun, keep_photons_beg=True, keep_photons_end=True, run_daq=False, max_steps=100)
            for ev in events:
                vert = ev.photons_beg.pos
                tracks = analyzer.generate_tracks(ev, qe=(1. / 3.))
                write_h5_reverse_track_file_event(f, vert, tracks, first)

                #vertices = utilities.AVF_analyze_event(analyzer, ev)
                #utilities.plot_vertices(ev.photons_beg.track_tree, 'AVF plot', reconstructed_vertices=vertices)
                if di_file is not None:
                    gun_specs = build_gun_specs(particle, lg, None, energy)
                    di_file = DIEventFile(config_name, gun_specs, ev.photons_beg.track_tree, tracks, ev.photons_beg)
                    di_file.write(di_file_base+'_'+str(i)+'.h5')

            print ('Photons detected: ' + str(tracks.sigmas.shape[0]))


def detector_config_from_parameter_array(config_name,
                                         config_dict,
                                         thickness_ratio=0.25,
                                         blockers=True,
                                         blocker_thickness_ratio=1.0/1000,
                                         lens_system_name=None,
                                         focal_length=1.0,
                                         light_confinement=True):

    return detectorconfig.DetectorConfig(
                config_dict[0],
                config_dict[1],
                config_dict[2],
                config_dict[3],
                0, 0, 1.0,        # pmtxbins, pmtybins, diameter_ratio
                thickness_ratio=thickness_ratio,
                blockers=blockers,
                blocker_thickness_ratio=blocker_thickness_ratio,
                lens_system_name=lens_system_name,
                focal_length=focal_length,
                EPD_ratio = config_dict[4],
                light_confinement=light_confinement,
                nsteps=config_dict[5],
                b_pixel=config_dict[6],
                tot_pixels=config_dict[7] if len(config_dict) > 7 else None,
                uuid=config_dict[8] if len(config_dict) > 8 else None,
                config_name=config_name)


def save_config_file(cfg, file_name, dict):
    config_path = paths.get_data_file_path(cfg)
    if not os.path.exists(config_path):
        os.makedirs(config_path)
    with open(config_path + file_name, 'w') as outf:
        pickle.dump(dict, outf)


def plot_vertices(track_tree, title, with_electrons=True, file_name=None, reconstructed_vertices=None, reconstructed_vertices2=None):
    particles = {}
    energies = {}

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax = fig.gca(projection='3d')

    if track_tree is not None:
        for key, value in track_tree.iteritems():
            if 'particle' in value:
                particle = value['particle']
                if particle not in particles:
                    particles[particle] = []
                    energies[particle] = []
                particles[particle].append(value['position'])
                energies[particle].append(100.*value['energy'])
        for key, value in particles.iteritems():
            if with_electrons or key != 'e-':
                the_array = np.array(value)
                #ax.plot(the_array[:,0], the_array[:,1], the_array[:,2], '.', markersize=5.0)
                ax.scatter(the_array[:,0], the_array[:,1], the_array[:,2], marker='o', s=energies[particle], label=key) #), markersize=5.0)
    if reconstructed_vertices is not None:
        vertex_positions = []
        for v in reconstructed_vertices:
            print(v.pos)
            vertex_positions.append(np.asarray(v.pos))
        vp = np.asarray(vertex_positions)
        print('AVF positions: ' + str(vp))
        ax.scatter(vp[:,0], vp[:,1], vp[:,2], marker=(6,1,0), s=100., color='gray', label='AVF') #), markersize=5.0)
    # Temporary
    if reconstructed_vertices2 is not None:
        vertex_positions = []
        for v in reconstructed_vertices2:
            print(v.pos)
            vertex_positions.append(np.asarray(v.pos))
        vp = np.asarray(vertex_positions)
        print('AVF 2 positions: ' + str(vp))
        ax.scatter(vp[:,0], vp[:,1], vp[:,2], marker=(6,1,0), s=100., color='black', label='AVF 2') #), markersize=5.0)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    plt.legend(loc=2)   # See https://pythonspot.com/3d-scatterplot/

    # See: http://fredborg-braedstrup.dk/blog/2014/10/10/saving-mpl-figures-using-pickle
    if file_name is not None:
        pickle.dump(fig, file(file_name, 'wb'))
    plt.show()

# Defaults for AVF
min_tracks = 0.1       # Use a fraction to take a fraction of total available tracks (something like that)
chiC = 0.75
temps = [256, 0.25]
tol = 0.1

# Doesn't do much
#### Note: AVF() modifies the tracks object ####
def AVF_analyze_tracks(analyzer, tracks, debug=False):
    vtcs = analyzer.AVF(tracks, min_tracks, chiC, temps, tol, debug)
    print('Vertices: ' + str(vtcs))
    return vtcs

# Not currently in use
def AVF_analyze_event(analyzer, event, debug=False):
    sig_cone = 0.01
    lens_dia = None
    n_ph = 0

    vtcs = analyzer.analyze_one_event_AVF(event, sig_cone, n_ph, min_tracks, chiC, temps, tol, debug, lens_dia)
    print('Vertices: ' + str(vtcs))
    return vtcs

def write_h5_reverse_track_file_event(h5file, vert, tracks, first):
    if first:
        en_depo = h5file.create_dataset('en_depo', maxshape=(None, 3), data=vert, chunks=True)
        h5file.create_dataset('coord', maxshape=(2, None, 3), data=[tracks.hit_pos.T, tracks.means.T], chunks=True)
        uncert = h5file.create_dataset('sigma', maxshape=(None,), data=tracks.sigmas, chunks=True)
        h5file.create_dataset('r_lens', data=tracks.lens_rad)

        h5file.create_dataset('idx_tr', maxshape=(None,), data=[uncert.shape[0]], chunks=True)
        h5file.create_dataset('idx_depo', maxshape=(None,), data=[en_depo.shape[0]], chunks=True)  # Need both maxshape and chunks True??
    else:
        en_depo = h5file.get('en_depo')      # This may be super inefficient??
        en_depo.resize(en_depo.shape[0] + vert.shape[0], axis=0)
        en_depo[-vert.shape[0]:, :] = vert
        coord = h5file.get('coord')
        coord.resize(coord.shape[1] + tracks.means.shape[1], axis=1)
        coord[:, -tracks.means.shape[1]:, :] = [tracks.hit_pos.T, tracks.means.T]
        uncert = h5file.get('sigma')
        uncert.resize(uncert.shape[0] + tracks.sigmas.shape[0], axis=0)
        uncert[-tracks.sigmas.shape[0]:] = tracks.sigmas

        # Untested - and is there a better way?  This looks too complicated
        idx_tr = h5file.get('idx_tr')
        idx_tr_size = idx_tr.shape[0]
        #print('=== ' + str(idx_tr_size))
        idx_tr.resize(idx_tr_size + 1, axis=0)
        idx_tr[idx_tr_size] = uncert.shape[0]

        idx_depo = h5file.get('idx_depo')
        idx_depo_size = idx_depo.shape[0]
        idx_depo.resize(idx_depo_size + 1, axis=0)
        idx_depo[idx_depo_size] = en_depo.shape[0]


def build_gun_specs(particle, position, momentum, energy):
    gs = dict(particle=particle, position=position, momentum=momentum, energy=energy)
    #gs = {'particle': particle, 'position': position, 'momentum': momentum, 'energy': energy}
    return gs

# A Distributed Imaging event file is a "deep dish" HDF5 file containing all of the data about this event
# Notes / TODO:
#   Need to add: config name, matrials config
#   There is currently much redundancy in the new hdf5 file format
#   Need to make this support mutiple events
#   Test new format without tracks
#   Cross check the config!!

'''
HDF5 file structure:
    config_name
    config: DetectorConfig object
    gun_specs
        particle
        position
        momentum
        energy
    track_tree: dict from chroma
    tracks:
    photons:

# Original HDF5 fields:
    hit_pos:
    means:
    sigmas:
'''


class DIEventFile(object):
    def __init__(self, config_name, gun_specs, track_tree, tracks, photons=None, full_event=None):
        self.config_name    = config_name
        self.gun_specs      = gun_specs
        self.track_tree     = track_tree
        self.tracks         = tracks
        self.photons        = photons
        self.full_event     = full_event

    @classmethod
    def load_from_file(cls, file_name):
        event = dd.io.load(file_name)

        config_name = event['config_name']
        gun_specs = event['gun']
        track_tree = event['track_tree']
        tracks = event['tracks']
        photons = event['photons']
        print('Photon count: ' + str(len(photons)))

        event_file = cls(config_name, gun_specs, track_tree, tracks, photons)
        event_file.full_event = event['full_event']

        # Preserve the whole thing in case we need access to 'hit_pos', 'means', 'sigmas' (for compatibility with the original HDF5 format)
        event_file.complete = event

        return event_file


    def write(self, file_name):
        event = {'track_tree': self.track_tree, 'gun': self.gun_specs, 'config_name': self.config_name}
        if self.config_name is not None:
            event['config'] = detectorconfig.configdict(self.config_name)
        if self.photons is not None:
            event['photons'] = self.photons
        if self.full_event is not None:
            event['full_event'] = self.full_event
        event['tracks'] = self.tracks
        event['hit_pos'] = self.tracks.hit_pos      # Note: these are the center of the lens that the photon hit
        event['means'] = self.tracks.means
        event['sigmas'] = self.tracks.sigmas
        print('Gun type: ' + str(type(self.gun_specs)))
        print('Writing deepdish file: ' + file_name)
        dd.io.save(file_name, event)

def print_tracks(tracks, count):
    print('Total track count: %d' % len(tracks))
    for index, track in enumerate(tracks):
        print('Track %d: %s, %s, %f, norm: %f' % (index, str(track[0]), str(track[1]), track[2], np.linalg.norm(track[0])))
        if index >= count:
            break

if __name__=='__main__':
    import DetectorResponseGaussAngle
    import EventAnalyzer

    parser = argparse.ArgumentParser()
    parser.add_argument('h5_file', help='Event HDF5 file')
    args = parser.parse_args()

    event = DIEventFile.load_from_file(args.h5_file)
    title = str(event.gun_specs['energy']) + ' MeV ' + str(event.gun_specs['particle'])     # 'str(particle)' in case it's None
    vertices = None
    if event.tracks is not None:
        print('Track count: ' + str(len(event.tracks)))
        cal_file = paths.get_calibration_file_name(event.config_name)
        print('Calibration file: ' + cal_file)

        print_tracks(event.tracks, 20)

        det_res = DetectorResponseGaussAngle.DetectorResponseGaussAngle(event.config_name, 10, 10, 10, cal_file)  # What are the 10s??
        '''
        tester_triangles = np.arange(1200)
        pixel_result = det_res.scaled_pmt_arr_surf(tester_triangles)
        for i in range(1200):
            print('%d\t%d\t%d\t%d\t%d' % (tester_triangles[i], pixel_result[0][i], pixel_result[1][i], pixel_result[2][i], pixel_result[3][i]))
        '''
        analyzer = EventAnalyzer.EventAnalyzer(det_res)
        vertices_from_original_run = AVF_analyze_tracks(analyzer, event.tracks, debug=True)
        '''
        print ("==================================================================")
        print ("==================================================================")
        for qe in [None]:  # 1./3.]: # , 1.0]:
            for i in range(1):
                new_tracks = analyzer.generate_tracks(event.full_event, qe=qe, debug=True)
                print_tracks(new_tracks, 20)
                new_vertices = AVF_analyze_tracks(analyzer, new_tracks)

                plot_vertices(event.track_tree, title + ', QE: ' + str(qe), reconstructed_vertices=vertices_from_original_run, reconstructed_vertices2=new_vertices)
        '''
        plot_vertices(event.track_tree, title, reconstructed_vertices=vertices_from_original_run)
