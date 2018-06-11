#
# utilities.py
# Contains utility methods to:
#       Manage detector configurations
#       Setup and run simulations
#       Plot vertices and tracks
#       Run AVF analyses
#       Create and load event files
#

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Required for projection='3d' below
import pickle
import numpy as np
import deepdish as dd
import argparse
import h5py
import os
import pprint
#import traceback

from chroma.detector import Detector
#from chroma.camera import view

import kabamland2 as kbl2
import detectorconfig
import lensmaterials as lm
import paths
from logger_lfd import logger

def load_or_build_detector(config, detector_material, g4_detector_parameters):
    configname = config.config_name
    filename_base = paths.detector_config_path + configname
    if not os.path.exists(paths.detector_config_path):
        os.makedirs(paths.detector_config_path)

    kabamland = None
    # How to ensure the material and detector parameters are correct??
    try:
        detector_config = dd.io.load(filename_base+'.h5')
        kabamland = detector_config['detector']
        logger.info("** Loaded HDF5 (deepdish) detector configuration: " + configname)
    except IOError as error:  # Will dd throw an exception?
        try:
            with open(filename_base+'.pickle','rb') as pickle_file:
                kabamland = pickle.load(pickle_file)
                logger.info("** Loaded pickle detector configuration: " + configname)
        except IOError as error:
            pass
    if kabamland is not None:
        config_has_g4_dp = hasattr(kabamland, 'g4_detector_parameters') and kabamland.g4_detector_parameters is not None
        config_has_g4_dm = hasattr(kabamland, 'detector_material') and kabamland.detector_material is not None
        if g4_detector_parameters is not None:
            logger.info('*** Using Geant4 detector parameters specified' +
                        (' - replacement' if config_has_g4_dp else '') + ' ***')
            kabamland.g4_detector_parameters = g4_detector_parameters
        elif config_has_g4_dp:
            logger.info('*** Using Geant4 detector parameters found in loaded file ***')
        else:
            logger.info('*** No Geant4 detector parameters found at all ***')

            if detector_material is not None:
                logger.info('*** Using Geant4 detector material specified' +
                            (' - replacement' if config_has_g4_dm else '') + ' ***')
                kabamland.detector_material = detector_material
            elif config_has_g4_dm:
                logger.info('*** Using Geant4 detector material found in loaded file ***')
            else:
                logger.info('*** No Geant4 detector material found at all ***')
    else:
        from chroma.loader import load_bvh  # Requires CUDA so only import it when necessary

        logger.info("** Building detector configuration: " + configname)
        kabamland = Detector(lm.create_scintillation_material(), g4_detector_parameters=g4_detector_parameters)
        kbl2.build_kabamland(kabamland, config)
        # view(kabamland)
        kabamland.flatten()
        kabamland.bvh = load_bvh(kabamland)
        try:
            with open(filename_base+'.pickle','wb') as pickle_file:
                pickle.dump(kabamland, pickle_file)
        except IOError as error:
            logger.info("Error writing pickle file: " + filename_base+'.pickle')

        # Write h5 file with configuration data structure
        logger.info('Saving h5 detector configuration.  UUID: %s' % config.uuid)

        '''   # This was created to minimize what is saved from the Detector object.  But for simplicity, we are currently pickling the whole object.
        detector_dict = {
            'detector_material' : kabamland.detector_material,
            'solids' : kabamland.solids,
            'solid_rotations' : kabamland.solid_rotations,
            'solid_displacements' : kabamland.solid_displacements,
            'bvh' : kabamland.bvh,
            'g4_detector_parameters' : kabamland.g4_detector_parameters,
            'solid_id_to_channel_index' : kabamland.solid_id_to_channel_index,
            'channel_index_to_solid_id' : kabamland.channel_index_to_solid_id,
            'channel_index_to_channel_id' : kabamland.channel_index_to_channel_id,
            'channel_id_to_channel_index' : kabamland.channel_id_to_channel_index,
            'time_cdf' : kabamland.time_cdf,
            'charge_cdf' : kabamland.charge_cdf
        }
        '''
        # TODO: Saving the whole dict and the object is redundant
        # TODO: Also, saving all of kabamland vs. just the parameters above adds about 1 Meg to the file size (I think)
        detector_data = {'config': config, 'config_dict': vars(config), 'detector': kabamland}
        dd.io.save(filename_base + '.h5', detector_data)

    return kabamland

# TODO: Better seed?
def sim_setup(config, in_file, useGeant4=False, geant4_processes=4, seed=12345, cuda_device=None, no_gpu=False):
    # Imports are here both to avoid loading Geant4 when unnecessary, and to avoid circular imports
    from chroma.detector import G4DetectorParameters
    import DetectorResponseGaussAngle
    import EventAnalyzer

    g4_detector_parameters = G4DetectorParameters(orb_radius=7., world_material='G4_Galactic') if useGeant4 else None
    detector = load_or_build_detector(config, lm.create_scintillation_material(), g4_detector_parameters=g4_detector_parameters)
    det_res = DetectorResponseGaussAngle.DetectorResponseGaussAngle(config,10,10,10,in_file)
    analyzer = EventAnalyzer.EventAnalyzer(det_res)
    if no_gpu:
        sim = None
        logger.warning('**** No GPU.  Not initializing CUDA or creating Simulation ****')
    else:
        from chroma.sim import Simulation
        sim = Simulation(detector, seed=seed, geant4_processes=geant4_processes if useGeant4 else 0, cuda_device=cuda_device)
    return sim, analyzer

# TODO: this currently also appears in nog4_sim.py
def sph_scatter(sample_count,in_shell,out_shell):
    logger.info('sph_scatter shell radii: ' + str(in_shell) + ' ' + str(out_shell))
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
def fire_g4_particles(sample_count, config, particle, energy, inner_radius, outer_radius, h5_file, location=None, momentum=None, di_file_base=None, qe=None):
    from chroma.generator import vertex

    config_name = config.config_name
    sim, analyzer = sim_setup(config, paths.get_calibration_file_name(config_name), useGeant4=True, geant4_processes=1, no_gpu=False)
    #analyzer.det_res.is_calibrated=False    # Temporary to test AVF with actual photon angles vs. calibration angles

    logger.info('Configuration:\t%s' % config_name)
    logger.info('Particle:\t\t%s ' % particle)
    logger.info('Energy:\t\t%d' % energy)
    logger.info('Sim count:\t\t%d' % sample_count)
    logger.info('File:\t\t%s' % h5_file)

    if location is None: # Location is a flag
        loc_array = sph_scatter(sample_count, inner_radius * 1000, outer_radius * 1000)
    else:
        loc_array = [location]

    with h5py.File(h5_file, 'w') as f:
        first = True
        logger.info('Running locations:\t%d' % len(loc_array))
        for i, lg in enumerate(loc_array):
            logger.info('Location:\t\t%s' % str(lg))
            if location is None:
                gun = vertex.particle_gun([particle], vertex.constant(lg), vertex.isotropic(), vertex.flat(float(energy) * 0.999, float(energy) * 1.001))
            else:
                gun = vertex.particle_gun([particle], vertex.constant(lg), vertex.constant(momentum), vertex.constant(energy)) # TODO: AWS seems to require: vertex.constant(np.array(momentum))

            events = sim.simulate(gun, keep_photons_beg=True, keep_photons_end=True, run_daq=False, max_steps=100)
            for ev in events:   # Note: There is really only ever one event because we enumerate loc_array above
                vert = ev.photons_beg.pos
                tracks = analyzer.generate_tracks(ev, qe=qe)
                write_h5_reverse_track_file_event(f, vert, tracks, first)
                first=False

                #vertices = utilities.AVF_analyze_event(analyzer, ev)
                #utilities.plot_vertices(ev.photons_beg.track_tree, 'AVF plot', reconstructed_vertices=vertices)
                if di_file_base is not None:
                    gun_specs = build_gun_specs(particle, lg, None, energy)
                    di_file = DIEventFile(config_name, gun_specs, ev.photons_beg.track_tree, tracks, ev.photons_beg, ev)
                    di_file.write(di_file_base+'_'+str(i)+'.h5')

            logger.info('Photons detected:\t%s' % str(tracks.sigmas.shape[0]))
            logger.info('============')

def plot_vertices(track_tree, title, with_electrons=True, file_name=None, reconstructed_vertices=None, reconstructed_vertices2=None):
    particles = {}
    energies = {}
    for key, value in track_tree.iteritems():
        if 'particle' in value:
            particle = value['particle']
            if particle not in particles:
                particles[particle] = []
                energies[particle] = []
            particles[particle].append(value['position'])
            energies[particle].append(100.*value['energy'])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for key, value in particles.iteritems():
        if with_electrons or key != 'e-':
            the_array = np.array(value)
            #ax.plot(the_array[:,0], the_array[:,1], the_array[:,2], '.', markersize=5.0)
            ax.scatter(the_array[:,0], the_array[:,1], the_array[:,2], marker='o', s=energies[key], label=key) #), markersize=5.0)
    if reconstructed_vertices is not None:
        vertex_positions = []
        for v in reconstructed_vertices:
            logger.info('Vertex position: %s' % v.pos)
            vertex_positions.append(np.asarray(v.pos))
        vp = np.asarray(vertex_positions)
        logger.info('AVF positions: ' + str(vp))
        ax.scatter(vp[:,0], vp[:,1], vp[:,2], marker=(6,1,0), s=100., color='gray', label='AVF') #), markersize=5.0)
    # Optionally plot reconstructed vertices as well
    if reconstructed_vertices2 is not None:
        vertex_positions = []
        for v in reconstructed_vertices2:
            logger.info('Vertex position: %s' % v.pos)
            vertex_positions.append(np.asarray(v.pos))
        vp = np.asarray(vertex_positions)
        logger.info('AVF 2 positions: ' + str(vp))
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

############
# This is the AVF call from efficiency.py:  (For reference)
#       eff_test(detfile,
#               detres=paths.get_calibration_file_name(detfile),
#               detbins=10,
#               sig_pos=0.01,
#               n_ph_sim=energy,
#               repetition=repetition,
#               max_rad=6600,
#               n_pos=n_pos,
#               loc1=(0,0,0),
#               sig_cone=0.01,
#               lens_dia=None,
#               n_ph=0,
#               min_tracks=0.1,
#               chiC=1.5,
#               temps=[256, 0.25],
#               tol=0.1,
#               debug=False)
############

# Defaults for AVF
min_tracks = 0.1       # Use a fraction of total event's tracks as the minimum # of tracks required for reconstruction
chiC = 0.75
temps = [256, 0.25]
tol = 0.1

# TODO: Could basically lose this function
#### Note: AVF() modifies the tracks object ####
def AVF_analyze_tracks(analyzer, tracks, debug=False):
    vtcs = analyzer.AVF(tracks, min_tracks, chiC, temps, tol, debug)
    logger.info('Vertices: ' + str(vtcs))
    return vtcs

# TODO: Not currently in use
def AVF_analyze_event(analyzer, event, debug=False):
    sig_cone = 0.01
    lens_dia = None
    n_ph = 0

    vtcs = analyzer.analyze_one_event_AVF(event, sig_cone, n_ph, min_tracks, chiC, temps, tol, debug, lens_dia)
    logger.info('Vertices: ' + str(vtcs))
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

        # TODO: rafactor into separate events rather than long lists with indices?
        idx_tr = h5file.get('idx_tr')
        idx_tr_size = idx_tr.shape[0]
        idx_tr.resize(idx_tr_size + 1, axis=0)
        idx_tr[idx_tr_size] = uncert.shape[0]

        idx_depo = h5file.get('idx_depo')
        idx_depo_size = idx_depo.shape[0]
        idx_depo.resize(idx_depo_size + 1, axis=0)
        idx_depo[idx_depo_size] = en_depo.shape[0]


def build_gun_specs(particle, position, momentum, energy):
    gs = dict(particle=particle, position=position, momentum=momentum, energy=energy)
    return gs

# A Distributed Imaging event file is a "deep dish" HDF5 file containing all of the data about this event
# TODO:
#   Need to add: materials config
#   Currently much redundancy in the  hdf5 file format
#   Make this support mutiple events
#   Test new format without tracks
#   Check the config UUID upon load or use

'''
HDF5 file structure:
    config_name
    config:         DetectorConfig object (pickled)
    gun_specs
        particle
        position
        momentum
        energy
    simulation_params:  TODO: not currently implemented
    track_tree:     dict from chroma (pickled)
    tracks:         output of EventAnalyzer.generate_tracks() (pickled)
    photons:        event.photons_beg (pickled)
    full_event:     full chroma event (pickled)

    # Original HDF5 file format fields:
    hit_pos:
    means:
    sigmas:
'''


class DIEventFile(object):
    def __init__(self, config_name, gun_specs, track_tree, tracks, photons=None, full_event=None, simulation_params=None):
        self.config_name    = config_name
        self.gun_specs      = gun_specs
        self.track_tree     = track_tree
        self.tracks         = tracks
        self.photons        = photons
        self.full_event     = full_event
        self.simulation_params = simulation_params

    @classmethod
    def load_from_file(cls, file_name):
        event = dd.io.load(file_name)

        config_name = event['config_name']
        gun_specs = event['gun']
        track_tree = event['track_tree']
        tracks = event['tracks']
        photons = event['photons']
        simulation_params = event['simulation_params'] if 'simulation_params' in event else None
        logger.info('Photon count: ' + str(len(photons)))

        event_file = cls(config_name, gun_specs, track_tree, tracks, photons, simulation_params=simulation_params)
        event_file.full_event = event['full_event']

        # Preserve the whole thing in case we need access to 'hit_pos', 'means', 'sigmas' (for compatibility with the original HDF5 format)
        event_file.complete = event
        return event_file

    def write(self, file_name):
        event = {'track_tree': self.track_tree, 'gun': self.gun_specs, 'config_name': self.config_name, 'simulation_params': self.simulation_params}
        if self.config_name is not None:
            event['config'] = detectorconfig.get_detector_config(self.config_name)
        if self.photons is not None:
            event['photons'] = self.photons
        if self.full_event is not None:
            event['full_event'] = self.full_event
        event['tracks'] = self.tracks
        if hasattr(self.tracks, 'hit_pos'):
            event['hit_pos'] = self.tracks.hit_pos      # Note: these are the center of the lens that the photon hit
            event['means'] = self.tracks.means
            event['sigmas'] = self.tracks.sigmas
        logger.info('Writing deepdish file: ' + file_name)
        dd.io.save(file_name, event)

def print_tracks(tracks, count):
    logger.info('Total track count: %d' % len(tracks))
    for index, track in enumerate(tracks):
        logger.info('Track %d: %s, %s, %f, norm: %f' % (index, str(track[0]), str(track[1]), track[2], np.linalg.norm(track[0])))
        if index >= count:
            break

# TODO: Stolen from EventAnalyzer -- integrate this back together
def plot_tracks_from_endpoints(begin_pos, end_pos, pts=None, highlight_pt=None, path=None, show=True, skip_interval=50, plot_title="Tracks"):
    # Returns a 3D plot of tracks (a Tracks object), as lines extending from their
    # PMT hit position to the inscribed diameter of the detector.
    # If pts is not None, will also draw them (should be a (3,n) numpy array).
    # If highlight_pt exists, it will be colored differently.
    # If path exists, a path will be drawn between its points (should be shape (3,n)).

    hit_pos = end_pos[0::skip_interval].T
    source_pos = begin_pos[0::skip_interval].T

    logger.info('Plotting %d tracks' % len(hit_pos[0]))

    xs = np.vstack((hit_pos[0, :], source_pos[0, :]))
    ys = np.vstack((hit_pos[1, :], source_pos[1, :]))
    zs = np.vstack((hit_pos[2, :], source_pos[2, :]))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Draw track hit positions
    ax.scatter(hit_pos[0, :], hit_pos[1, :], hit_pos[2, :], color='red')
    # Draw tracks as lines
    for ii in range(len(hit_pos[0])):
        ax.plot(xs[:, ii], ys[:, ii], zs[:, ii], color='red')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(plot_title)

    # Draw pts
    if pts is not None:
        ax.scatter(pts[0, :], pts[1, :], pts[2, :], color='blue')

    # Draw highlight_pt, larger and different color
    if highlight_pt is not None:
        ax.scatter(highlight_pt[0], highlight_pt[1], highlight_pt[2], color='green', s=50)

    # Draw path between points in path
    if path is not None:
        ax.plot(path[0, :], path[1, :], path[2, :], color='blue')
        plt.title('Vertex position record')

    if show:
        plt.show()

    return fig

# Driver for generating new hdf5/dd event files and diagnosing AVF algorithm
if __name__=='__main__':
    import DetectorResponseGaussAngle
    import EventAnalyzer

    parser = argparse.ArgumentParser()
    parser.add_argument('h5_file', help='Event HDF5 file')
    args = parser.parse_args()

    event = DIEventFile.load_from_file(args.h5_file)
    title = str(event.gun_specs['energy']) + ' MeV ' + str(event.gun_specs['particle'])
    vertices = None
    if event.tracks is not None:
        logger.info('Track count: ' + str(len(event.tracks)))
        #event.tracks.sigmas.fill(0.01)  # TODO: Temporary hack because I think we forced 0.0001 into the tracks in the test file.  Sigmas too small really screw up the machine!!
        print_tracks(event.tracks, 20)

        calibrated_simulation = True
        if event.simulation_params is not None and 'calibrated' in event.simulation_params:
            calibrated_simulation = event.simulation_params['calibrated']
            logger.info('=== Simulation event used calibrated detector: %s ===' % str(calibrated_simulation))
        else:
            logger.info('=== No calibration flag in event file ===')
            logger.info('Photons in file: %d' % len(event.full_event.photons_end))

        config = detectorconfig.get_detector_config(event.config_name)  # Cross check UUID

        if calibrated_simulation:
            cal_file = paths.get_calibration_file_name(event.config_name)
            logger.info('Calibration file: ' + cal_file)
            det_res = DetectorResponseGaussAngle.DetectorResponseGaussAngle(config, 10, 10, 10, cal_file)  # What are the 10s??
        else:
            det_res = DetectorResponseGaussAngle.DetectorResponseGaussAngle(config, 10, 10, 10)   # What are the 10s??

        logger.info('Tracks in file: %d' % len(event.tracks))
        analyzer = EventAnalyzer.EventAnalyzer(det_res)
        # analyzer.plot_tracks(event.tracks)
        logger.info('=== Analyzing tracks from event tracks in file ===')
        vertices_from_original_run = AVF_analyze_tracks(analyzer, event.tracks, debug=True)

        if event.track_tree is not None:
            plot_vertices(event.track_tree, title, reconstructed_vertices=vertices_from_original_run)

        # Why aren't these tracks the same as the tracks in the event in the file?
        new_tracks = analyzer.generate_tracks(event.full_event, qe=None, debug=True)
        print_tracks(new_tracks, 20)
        analyzer.plot_tracks(new_tracks)

        if calibrated_simulation:

            '''
            tester_triangles = np.arange(1200)
            pixel_result = det_res.scaled_pmt_arr_surf(tester_triangles)
            for i in range(1200):
                logger.info('%d\t%d\t%d\t%d\t%d' % (tester_triangles[i], pixel_result[0][i], pixel_result[1][i], pixel_result[2][i], pixel_result[3][i]))
            '''
            '''
            logger.info("==================================================================")
            logger.info("==================================================================")
            for qe in [None]:  # 1./3.]: # , 1.0]:
                for i in range(1):
                    new_tracks = analyzer.generate_tracks(event.full_event, qe=qe, debug=True)
                    print_tracks(new_tracks, 20)
                    new_vertices = AVF_analyze_tracks(analyzer, new_tracks)

                    plot_vertices(event.track_tree, title + ', QE: ' + str(qe), reconstructed_vertices=vertices_from_original_run, reconstructed_vertices2=new_vertices)
            '''
        else:

            #plot_tracks_from_endpoints(event.full_event.photons_beg.pos, event.full_event.photons_end.pos, skip_interval=150, plot_title="All photon tracks")

            # Why aren't these tracks the same as the tracks in the event in the file?
            #new_tracks = analyzer.generate_tracks(event.full_event, qe=None, debug=True)
            #print_tracks(new_tracks, 20)
            #analyzer.plot_tracks(new_tracks)

            #vertices_from_file_tracks = AVF_analyze_tracks(analyzer, new_tracks, debug=True)

            #plot_vertices(event.track_tree, title) # , reconstructed_vertices=vertices_from_original_run)
            pass
