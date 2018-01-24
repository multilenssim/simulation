import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Required for projection='3d' below
import pickle
import numpy as np
import deepdish as dd
import argparse

import detectorconfig
import DetectorResponseGaussAngle
import EventAnalyzer
import paths

def plot_vertices(track_tree, title, with_electrons=True, file_name='vertex_plot.pickle', reconstructed_vertices=None):
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
    if reconstructed_vertices is not None:
        vertex_positions = []
        for v in reconstructed_vertices:
            print(v.pos)
            vertex_positions.append(np.asarray(v.pos))
        vp = np.asarray(vertex_positions)
        print('AVF positions: ' + str(vp))
        ax.scatter(vp[:,0], vp[:,1], vp[:,2], marker=(6,1,0), s=100., color='gray', label='AVF') #), markersize=5.0)

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

def AVF_analyze_tracks(analyzer, tracks):
    min_tracks = 0.1
    chiC = 1.5
    temps = [256, 0.25]
    tol = 0.1
    debug = True

    vtcs = analyzer.AVF(tracks, min_tracks, chiC, temps, tol, debug)
    print('Vertices: ' + str(vtcs))
    return vtcs

def AVF_analyze_event(analyzer, event):
    sig_cone = 0.01
    lens_dia = None
    n_ph = 0
    min_tracks = 0.1
    chiC = 1.5
    temps = [256, 0.25]
    tol = 0.1
    debug = True

    vtcs = analyzer.analyze_one_event_AVF(event, sig_cone, n_ph, min_tracks, chiC, temps, tol, debug, lens_dia)
    print('Vertices: ' + str(vtcs))
    return vtcs

def build_gun_specs(particle, position, momentum, energy):
    gs = dict(particle=particle, position=position, momentum=momentum, energy=energy)
    #gs = {'particle': particle, 'position': position, 'momentum': momentum, 'energy': energy}
    return gs

# Write a "deep dish" HDF5 file containing all of the data about this event
# Need to add: config name, matrials config
def write_deep_dish_file(file_name, config_name, gun_specs, track_tree, tracks, photons=None):
    data = {'track_tree': track_tree, 'gun': gun_specs, 'config_name': config_name}
    #data['photon_positions'] = output.pos
    if config_name is not None:
        data['config'] = detectorconfig.configdict(config_name)
    if photons is not None:
        data['photons'] = photons
    data['tracks'] = tracks
    data['hit_pos'] = tracks.hit_pos
    data['means'] = tracks.means
    data['sigmas'] = tracks.sigmas
    print('Gun type: ' + str(type(gun_specs)))
    print('Writing deepdish file: ' + file_name)
    dd.io.save(file_name, data)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('h5_file', help='Event HDF5 file')
    args = parser.parse_args()
    event = dd.io.load(args.h5_file)
    # Note - there is currently a lot of redundancy in the new hdf5 file format

    vertices = None
    if event['tracks'] is not None:
        config_name = event['config-name']
        det_res = DetectorResponseGaussAngle(config_name, 10, 10, 10, paths.get_calibration_file_name(config_name))       # What are the 10s??
        analyzer = EventAnalyzer(det_res)
        vertices = AVF_analyze_tracks(analyzer, event['tracks'])

    # Test this without tracks
    plot_vertices(event['track_tree'], "Woo hoo!", reconstructed_vertices=vertices)
