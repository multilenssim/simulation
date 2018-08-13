import time
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pprint

import paths
#from EventAnalyzer import EventAnalyzer
import utilities
from logger_lfd import logger
import detectorconfig

from chroma.sample import uniform_sphere
from chroma.event import Photons

def create_double_fixed_source_events(loc1, loc2, amount1, amount2):
    import kabamland2 as kbl

    # produces a list of Photons objects, each with two different photon sources at fixed locations
    events = []
    # Move constant photons etc. to driver utils
    events.append(kbl.constant_photons(loc1, int(amount1)) + kbl.constant_photons(loc2, int(amount2)))
    return events

def create_coordinate_segments(start, stop, count):
    #logger.info('Start: %s, stop: %s, count: %s' % (str(start), str(stop), count))
    if start == stop:
        return [start]*count
    else:
        array = np.arange(start, stop, (stop - start)/float(count), dtype=float)
        if len(array) != count:  # Might just align to give one too many - fix that
            return array[:-1]    # This may not be quite correct - need something that guarantees that the points are on the line
        return array

# Line is a tuple of 2 endpoints
# TODO: let's turn this into a generator?
def line_of_photons(line, direction, count):
    xs = create_coordinate_segments(line[0][0], line[1][0], count)
    ys = create_coordinate_segments(line[0][1], line[1][1], count)
    zs = create_coordinate_segments(line[0][2], line[1][2], count)
    #logger.info('Zs: %s' % zs)
    positions = np.dstack((xs,ys,zs))
    #logger.info('Positions: %s' % str(positions[0]))  # Need to figure out why we need the zero??
    pol = np.cross(direction, uniform_sphere(count)) # ????
    dirvec = [direction,]*count
    #logger.info('Shapes: %s %s' % (np.shape(positions), np.shape(dirvec)))
    #logger.info('Direction vector:  %s' % str(dirvec))
    #300 nm is roughly the pseudocumene scintillation wavelength
    wavelengths = np.repeat(300.0, count)  # Do we need the repeat?   YES - I think so...
    return Photons(positions[0], dirvec, pol, wavelengths)  # Direction will probably need to be an array

LENS_TRIANGLE_COUNT = 7940

def find_line_parallel_to_lens(config, length, lens):
    lens_vertex = config.vtx[lens]      # Just an arbitrary one
    # Line will start from (0,0,0)
    random_parallel_vector = np.cross(lens_vertex, (0,0,1))   # Will be perpendicular to the ray from the origin to the lens center
    scaled_rpv = random_parallel_vector / np.linalg.norm(random_parallel_vector) * length
    logger.info('Line direction: %s' % str(scaled_rpv))
    return scaled_rpv

def plot_lens(ax, detector, lens):
    vertices = detector.mesh.vertices
    offset = lens * LENS_TRIANGLE_COUNT
    ax.plot(vertices[offset: offset+LENS_TRIANGLE_COUNT, 0],
            vertices[offset: offset+LENS_TRIANGLE_COUNT, 1],
            vertices[offset: offset+LENS_TRIANGLE_COUNT, 2], '.', markersize=1, color="green")
    #ax.plot(vtx[:, 0], vtx[:, 1], vtx[:, 2], '.')  # Plot all lenses

def plot_sphere(ax, radius):
    phi, theta = np.mgrid[0.0:np.pi:20j, 0.0:2.0 * np.pi:20j]
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    ax.plot_surface(x, y, z, rstride=1, cstride=1, color='c', alpha=0.1, linewidth=0)


# Started from EventAnalyzer
def plot_event_photons(event, detector, lens, intermediate_points=None):
    if intermediate_points is not None:
        xs = np.vstack((event.photons_beg.pos[:, 0], intermediate_points[:, 0]))
        ys = np.vstack((event.photons_beg.pos[:, 1], intermediate_points[:, 1]))
        zs = np.vstack((event.photons_beg.pos[:, 2], intermediate_points[:, 2]))
        x2s = np.vstack((intermediate_points[:, 0], event.photons_end.pos[:,0]))
        y2s = np.vstack((intermediate_points[:, 1], event.photons_end.pos[:,1]))
        z2s = np.vstack((intermediate_points[:, 2], event.photons_end.pos[:,2]))
    else:
        xs = np.vstack((event.photons_beg.pos[:,0], event.photons_end.pos[:,0]))
        ys = np.vstack((event.photons_beg.pos[:,1], event.photons_end.pos[:,1]))
        zs = np.vstack((event.photons_beg.pos[:,2], event.photons_end.pos[:,2]))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Draw tracks as lines
    for ii in range(len(event.photons_beg)):
        ax.plot(xs[:, ii], ys[:, ii], zs[:, ii], color='red')
    if intermediate_points is not None:
        for ii in range(len(event.photons_beg)):
            ax.plot(x2s[:, ii], y2s[:, ii], z2s[:, ii], color='blue')

    plot_lens(ax, detector, lens)
    plot_sphere(ax, 7556)  # Get rid of this magic number

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # plt.title(plot_title)

    plt.show()

def detector_test(config, sim, analyzer, lens, cutoff=0.5):
    step = 500.
    photon_count = 50
    offset_count = 12
    line_dir = find_line_parallel_to_lens(config, config.half_EPD, lens)

    # To make the "butterfly"
    '''
    sim_events = [None] * 2 * offset_count
    for i in range(0, 2 * offset_count, 2):
        n_meter_offset = i/2 * step * line_dir / np.linalg.norm(line_dir)
        sim_events[i] = line_of_photons(((0, 0, 0)+n_meter_offset, line_dir+n_meter_offset), (-config.vtx[lens])-((0, 0, 0)+n_meter_offset), photon_count)
        sim_events[i+1] = line_of_photons(((0, 0, 0)-n_meter_offset, -line_dir-n_meter_offset), (-config.vtx[lens])-((0, 0, 0)-n_meter_offset), photon_count)
    '''
    sim_events = [None]  * offset_count
    for i in range(0, offset_count):
        n_meter_offset = i * step * line_dir / np.linalg.norm(line_dir)
        sim_events[i] = line_of_photons(((0, 0, 0)+n_meter_offset, line_dir+n_meter_offset), (-config.vtx[lens])-((0, 0, 0)+n_meter_offset), photon_count)

    logger.info('Vertex: %s, length: %f' % (config.vtx[lens], np.linalg.norm(config.vtx[lens])))

    angles = []
    angles2 = []
    results = []
    results2 = []
    i = 0
    last_photon_count = 0
    detector_center = None
    for ev in sim.simulate(sim_events, keep_photons_beg=True, keep_photons_end=True, run_daq=False, max_steps=100):
        tracks = analyzer.generate_tracks(ev)   # Just to get output of rings etc
        ii = 0
        first = True
        for start, end in zip(ev.photons_beg.pos, ev.photons_end.pos):
            if first and i == 0:   # HAve to assume that the first photon is not reflected
                detector_center = end  # This is a bit error prone - explicitly fire a single photon
            dist = np.linalg.norm(end - detector_center)
            if dist < (config.half_EPD * 1.5):  # Drop reflections
                if first:  # Save the first photon
                    # Butterfly: angle = (1 if (i % 2 == 0) else -1) * np.arcsin(np.linalg.norm(start) / np.linalg.norm(end))
                    angle = np.arcsin(np.linalg.norm(start) / np.linalg.norm(end))
                    print('i: %d, angle: %f' % (i, angle))
                    first = False
                if ii < (cutoff * photon_count):
                    results.append(dist)
                    angles.append(angle)
                else:
                    results2.append(dist)
                    angles2.append(angle)
            else:
                print('Out of bounds distance: %f' % dist)
            ii += 1
        offset = i * step * line_dir / np.linalg.norm(line_dir)
        intermediate_points = ev.photons_beg.pos - config.vtx[lens] - offset
        plot_event_photons(ev, sim.detector, lens, intermediate_points)
        current_count = len(results) + len(results2)
        print('Photon count at this offset: %d' % (current_count - last_photon_count))
        last_photon_count = current_count
        i += 1
    plt.scatter(angles2,results2, s=40, marker='+', label=str(cutoff) +  ' EPD < pos. < 1.0 EPD', color='red', alpha=0.5)
    plt.scatter(angles,results, s=4.,label='0 < pos. < ' + str(cutoff)  + ' EPD', color='blue', alpha=0.4)

    # Add trendline for < 0.5 EPD
    # Really want to compute an average of the ponts at each angle outselves - not a curve fit
    trend = np.polyfit(angles, results, 3)
    p = np.poly1d(trend)
    plt.plot(angles, p(angles), "b--", linewidth=0.5)

    plt.grid(axis='y', linestyle='-', linewidth=0.5)
    plt.xlabel("Angle of incidence on lens (radians)")
    plt.ylabel("Linear distance from detector center (mm)")
    plt.title('Lens #%d, Configuration: %s' % (lens, config.config_name))
    plt.legend(loc=4)
    plt.show()


# Pass in sim and analyzer so don't have to recreate them each time (performace optimization)
# Do we need to pass in both?
def run_simulation_double_fixed_source(sim, analyzer, sample, config, loc1, loc2, amount, qe=None):
    import h5py
    import numpy as np

    config_name = config.config_name

    # File pathing stuff should not be in here
    data_file_dir = paths.get_data_file_path_no_raw(config_name)
    if not os.path.exists(data_file_dir):
        os.makedirs(data_file_dir)
    dist = np.linalg.norm(loc2 - loc1)
    fname_base = data_file_dir + 'd-site-'+str(int(dist/10))+'cm-'+str(qe)
    if UNCALIBRATE:
        fname_base += '-uncal'
    fname = fname_base + '.h5'

    logger.info('Firing %d photons from %s and %s' % (amount, str(loc1), str(loc2)))
    logger.info('Configuration loaded: %s' % config_name)
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
                utilities.write_h5_reverse_track_file_event(f, vert, tracks, first)
                # Plot ring histogram
                if not UNCALIBRATE:
                    _,bn,_ = plt.hist(tracks.rings,bins=100)
                    #plt.yscale('log', nonposy='clip')
                    plt.xlabel('ring')
                    plt.show()

                #vertices = utilities.AVF_analyze_tracks(analyzer, tracks, debug=True)

                #vertices = utilities.AVF_analyze_event(analyzer, ev, debug=True)
                #utilities.plot_vertices(ev.photons_beg.track_tree, 'AVF plot', reconstructed_vertices=vertices)

                gun_specs = utilities.build_gun_specs(None, loc1, None, amount)      # TODO: Need loc2?? & using amount of photons as energy
                di_file = utilities.DIEventFile(config_name,
                                                gun_specs,
                                                ev.photons_beg.track_tree,
                                                tracks,
                                                photons=ev.photons_beg,
                                                full_event=ev,
                                                simulation_params={'calibrated': (not UNCALIBRATE)}   # TODO: Just a quick hack for now
                )
                di_file.write(fname_base + '_' + str(i) + '.h5')

                first = False
                i += 1

            logger.info('Time: ' + str(time.time() - start))

CEE = 300. # Speed of light in mm / nanoseconds

def plot_photon_times(photons, origin=None):
    if origin is not None:
        distances = np.linalg.norm(photons.pos - origin, axis=1)
        times = photons.t - distances / CEE
    else:
        times = photons.t
    plt.hist(times, bins=200, range = [0, 10.])
    plt.yscale('log')
    plt.grid(True)
    plt.show()


def analyze_neutron_events(track_trees, events, neutron_energy):
    print('neutron energy\t particle 1 \t number \t energy \t time \t particle 2 \t number \t energy \t time \t computed distance \t G4 distance')
    p2_energies = []
    p2_times = []
    p3_energies = []
    p3_times = []
    computed_distances = []
    G4_distances = []
    event_centers = []
    for index, track_tree in enumerate(track_trees):
        neutron = track_tree[1]
        if neutron['particle'] != 'neutron':
            print('Uh oh!  First particle is a ' + neutron['particle'])
            neutron_energy = np.nan
        else:
            neutron_energy = neutron['energy']
        particle_numbers = sorted(track_tree.keys())
        p2 = track_tree[particle_numbers[2]]
        p3 = track_tree[particle_numbers[3]]
        # Per Jacopo - take all particles
        # if p2['particle'] == 'proton' and p3['particle'] == 'proton':
        if p3['particle'] != 'deuteron':        # I believe that this is excluding the capture
            p2_energies.append(p2['energy'])
            p2_times.append(p2['time'])
            event_centers.append(p2['position'])      # Use the Grant4 position for now.  Will have to reconstruct this eventually
            p3_energies.append(p3['energy'])
            p3_times.append(p3['time'])
            computed_distance = pow(2. * (neutron_energy - p2['energy']) / pow(10.,3.), 0.5) * (p3['time'] - p2['time']) * CEE
            computed_distances.append(computed_distance)
            G4_distance = np.linalg.norm(np.asarray(p3['position']) - np.asarray(p2['position']))
            G4_distances.append(G4_distance)
            print('%f\t%s\t%d\t%f\t%f\t%s\t%d\t%f\t%f\t%f\t%f' % (neutron_energy,
                                                  p2['particle'], particle_numbers[2], p2['energy'], p2['time'],
                                                  p3['particle'], particle_numbers[3], p3['energy'], p3['time'],
                                                  computed_distance,
                                                  G4_distance))

        #else:
        #    print('Uh oh!  At least one of the first two particles is something other than a proton: %s, %s' % (p2['particle'], p3['particle']))

    fig, axs = plt.subplots(nrows=2, ncols=3)  # , sharey=True)

    fig.suptitle('Neutron scatter statistics - %.1f MeV' % neutron_energy, fontsize=18)  # TODO: Using the last neutron energy is a complete hack

    subplot = axs[0][0]
    _, _, _ = subplot.hist(p2_energies, bins=25)
    subplot.set_xlabel('Energy (MeV)', fontsize=10)
    #subplot.set_yscale('', nonposy='clip')
    subplot.set_title('First proton/particle', fontsize=14)

    subplot = axs[0][1]
    _, _, _ = subplot.hist(p2_times, bins=25)
    subplot.set_xlabel('Time (ns)', fontsize=10)
    #subplot.set_yscale('', nonposy='clip')
    subplot.set_title('First proton/particle', fontsize=14)

    subplot = axs[0][2]
    _, _, _ = subplot.hist(computed_distances, bins=25)
    subplot.set_xlabel('Dist (mm)', fontsize=10)
    #subplot.set_yscale('', nonposy='clip')
    subplot.set_title('Distance between particles', fontsize=14)

    subplot = axs[1][0]
    _, _, _ = subplot.hist(p3_energies, bins=25)
    subplot.set_xlabel('Energy (MeV)', fontsize=10)
    #subplot.set_yscale('', nonposy='clip')
    subplot.set_title('Second proton/particle', fontsize=14)

    subplot = axs[1][1]
    _, _, _ = subplot.hist(p3_times, bins=25)
    subplot.set_xlabel('Time (ns)', fontsize=10)
    #subplot.set_yscale('', nonposy='clip')
    subplot.set_title('Second proton/particle', fontsize=14)

    # See: https://matplotlib.org/examples/statistics/histogram_demo_cumulative.html
    subplot = axs[1][2]
    mu = 200
    sigma = 25
    n_bins = 50

    # plot the cumulative histogram
    n, bins, patches = subplot.hist(computed_distances, n_bins, normed=1, histtype='step', cumulative=True, label='Empirical')

    # Overlay a reversed cumulative histogram.
    subplot.hist(computed_distances, bins=bins, normed=1, histtype='step', cumulative=-1, label='Reversed emp.')

    subplot.set_xlabel('Dist (mm)', fontsize=10)
    #subplot.set_yscale('', nonposy='clip')
    subplot.set_title('Distance between particles', fontsize=14)

    # See this for plotting a curve fit: https://plot.ly/matplotlib/polynomial-fits/

    # Add a line showing the expected distribution.
    #y = mlab.normpdf(bins, mu, sigma).cumsum()
    #y /= y[-1]
    # ax.plot(bins, y, 'k--', linewidth=1.5, label='Theoretical')

    # tidy up the figureq
    subplot.grid(True)
    #subplot.legend(loc='right')
    #subplot.set_title('Cumulative step histograms')
    #subplot.set_xlabel('Annual rainfall (mm)')
    #subplot.set_ylabel('Likelihood of occurrence')

    # plt1.grid(True)# See: https://matplotlib.org/examples/pylab_examples/axes_props.html
    #fig.set_size_inches(11.5, 14)
    # plt.tight_layout()  # Doesn't help
    plt.subplots_adjust(hspace=0.45)
    #fig.savefig(config.config_name + '-calsim-plot.pdf', bbox_inches='tight', pad_inches=0)  # save the figure to file
    #plt.close(fig)
    plt.show()
    return event_centers

def display_track_tree(track_tree):
    for key, value in track_tree.iteritems():
        if value.get('particle') is not None:  # Ignore the first entry
            print('Particle: %s, Energy: %f, Time: %.02f, Processed: %s' % (
            value['particle'], value['energy'], value['time'], value['child_processes'],))


UNCALIBRATE = False

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_name', help='detector configuration', nargs='?', default='cfSam1_l200_p107600_b4_e10')
    parser.add_argument('particle', help='particle to simulate')
    parser.add_argument('s_d', help='seed location', nargs='?', default='01')
    _args = parser.parse_args()

    inner_radius = int(_args.s_d[0])
    outer_radius = int(_args.s_d[1])
    seed_loc = 'r%i-%i' % (inner_radius, outer_radius)

    particle = _args.particle
    config_name = _args.config_name

    data_file_dir = paths.get_data_file_path_no_raw(config_name)
    if not os.path.exists(data_file_dir):
        os.makedirs(data_file_dir)

    # Debugging only - shouldn't need Geant4 if we are just firing photons
    import Geant4
    logger.info('G4 state: %s' % Geant4.gStateManager.GetCurrentState())
    logger.info('G4 Random engine: %s' % Geant4.HepRandom.getTheEngine())
    logger.info('G4 Random seed: %s' % Geant4.HepRandom.getTheSeed())

    config = detectorconfig.get_detector_config(config_name)

    #for particle in ['neutron']: # ['e-']:  # ,'gamma']:
    #for dist_range in ['01']:  #,'34']:
    _sample_count = 2

    aggregate_photons_times = []

    if particle != 'photon':
        # Pass sim, analyzer in to avoid reloading the detector and impr         ove improve performance
        #sim, analyzer = utilities.sim_setup(config, paths.get_calibration_file_name(config_name), useGeant4=False, geant4_processes=1)
        for energy in [5.]: # ,10,50]:
            qe=None
            fname_base = data_file_dir+seed_loc+'_'+str(energy)+'_'+particle+'_'+str(qe)+'_sim'
            fname = fname_base+'.h5'
            track_trees, events = utilities.fire_g4_particles(_sample_count, config, particle, energy,
                                        inner_radius, outer_radius, fname, qe=qe,
                                        momentum=np.array([1., 0., 0.]), time_cut=500)    # What effect does momentum have?
            first_event = None
            for event in events:
                aggregate_photons_times.extend(event.photons_end.t)
                if first_event is None:
                    first_event = event
            display_track_tree(track_trees[0])
            event_centers = analyze_neutron_events(track_trees, events, energy)
            #plot_photon_times(aggregate_photons_times, event_centers[0])  # Just pick the first one for now.  Make this handle multiple events later
            plot_photon_times(first_event.photons_beg) # , event_centers[0])


            '''
            utilities.fire_g4_particles(_sample_count, config, particle, energy,
                                           inner_radius, outer_radius, fname, di_file_base=fname_base, qe=qe,
                                           location=np.array([0.,0.,0.]), momentum=np.array([1.,0.,0.]))
            '''
    else:  # Photons only - need to clean this up - what optiona?  Single vs. double site?
        '''lens = kb.get_lens_triangle_centers(config.vtx, config.half_EPD, config.diameter_ratio, lens_system_name=config.lens_system_name)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(lens[:,0], lens[:,1], lens[:,2], '.')
        plt.show()'''


        if UNCALIBRATE:
            logger.info('==== Forcing an uncalibrated detector ====')
        for dist in [1]: # [100., 500.,1000.,1500.,2000.]:
            sim, analyzer = utilities.sim_setup(config, paths.get_calibration_file_name(config_name), useGeant4=False)
            if UNCALIBRATE:
                analyzer.det_res.is_calibrated = False    # Temporary to test AVF with actual photon angles

            # Test lens aberration
            for i in [60,61,62,63,64,65,66,67,68,100]: # [0,1,10,76,138]:
                detector_test(config, sim, analyzer, i)

            #run_simulation_double_fixed_source(sim, analyzer, _sample_count, config, np.array([0.,0.,0.]), np.array([dist,0.,0.]), 16000, qe=None)
