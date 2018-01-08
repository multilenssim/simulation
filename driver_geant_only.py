# Test program for creating scintillation (and other) photons from Geant4

import os
# These 'environment variables' are not passed in.  They show up in 'set' but not 'env' on MacOS
os.environ['LD_LIBRARY_PATH'] = '~/Development/physics/chroma_env/lib'
os.environ['DYLD_LIBRARY_PATH'] = '~/Development/physics/chroma_env/lib'

from chroma.event import Vertex
from chroma.generator import g4gen
from chroma.detector import G4DetectorParameters

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
import pprint
import math
import pickle

import Geant4		# Only needed to turn logging on
from Geant4.hepunit import *
from Geant4 import *

import lensmaterials
import count_processes

from mpl_toolkits.mplot3d import Axes3D

class G4Generator:
    def generate(self, particle_name, position, direction, scintillator, generator, energy=2.):
        #from chroma.generator import g4gen
        gen = generator  # g4gen.G4Generator(scintillator)
        vertex = Vertex(particle_name, position, direction, energy)
        output = gen.generate_photons([vertex], mute=False)
        # print(vertex.particle_name + " photons from/direction/energy/count:\t" +
        #	  str(vertex.pos) + '\t' + str(vertex.dir) + '\t' + str(vertex.ke) + '\t' + str(len(output.pos)))

        '''
        print("====>>>> Unloading g4gen")
        # Not sure if have to do both of these
        # See: https://stackoverflow.com/questions/437589/how-do-i-unload-reload-a-python-module
        del sys.modules["chroma.generator.g4gen"]
        del g4gen
        # reload(g4gen)
        '''
        return output

    #def __del__(self):
        #print("========>>>>>>>> Destructing G4Generator instance")
        # super.__del__(self)  No call to super needed ???  It raises an error...

def compute_stats(data, x_distances):
    average = np.empty([len(data)], dtype='float')
    error = np.empty([len(data)], dtype='float')
    count = 0
    for x in x_distances:
        list = data[x]
        average[count] = np.average(list)
        error[count] = np.std(list)
        count += 1
    # We could glue all the arrays together and then not have to loop over each vector of runs: (currently they are in dictionaries
    '''
    the_array = []
    for x in x_distances:
        the_array.append(data[x])
    average2 = np.average(the_array, axis=1)
    '''
    return average, error

def fire_particles_old(particles, run_count, energy):
    scintillator = lensmaterials.create_scintillation_material()

    x_position = None
    if len(sys.argv) > 1:
        x_position = float(sys.argv[1])

    g4_params = G4DetectorParameters(world_material='G4_AIR', orb_radius=7.)
    gen = g4gen.G4Generator(scintillator, g4_detector_parameters=g4_params)
    momentum = (1, 0, 0)

    # Wierd things happen when we make the world out of scintillator
    # Geant4 must be adding the orb material to the scintillator?
    # But even that doesn't explain everything (i.e. we get somewhat the expected material with air in the orb, but not with Galactic??
    #gen = g4gen.G4Generator("G4_Galactic", orb_radius=7., world_material=scintillator)
    #momentum = (-1, 0, 0)

    # Old code:
    '''
    if x_position is not None:
        position = (x_position * m, 0., 0.)
    else:
        position = (0 * m, 0., 0.)
    g4 = G4Generator()
    out_ph1 = g4.generate('e-', position, momentum, scintillator, gen, energy=2.)		# NOTE the energy here.  for testing !!!!!
    g4 = None
    '''

    for particle in gParticleTable.GetParticleList():
        particle.DumpTable()    # This duplicates the DumpTable() above
        if particle.GetParticleName() in particles:
            # particle.DumpTable()
            pm = particle.GetProcessManager()
            print('===>>> Particle process manager: ' +  str(particle.GetParticleName()) + ' <<<===')
            pm.DumpInfo()


    e_x_distances = [0.] # np.linspace(6.97, 7.04, 29)    # (6.97, 7.04, 29)		# (6.99, 7.001, 12)
    gamma_x_distances = [0.] # np.linspace(5.5, 7.20, 35)    # (5.5, 7.20, 35)					# (5.8, 7.05, 26)

    counts = {}
    scint_counts = {}
    cherenkov_counts = {}
    track_counts = {}
    vertex_positions = {}

    for particle in particles:
        counts[particle] = {}
        scint_counts[particle] = {}
        cherenkov_counts[particle] = {}
        track_counts[particle] = np.ndarray(run_count, dtype=int)
        vertex_positions[particle] = np.ndarray(run_count, dtype=float)

    for counter in xrange(run_count):
        for particle in particles:
            print("===> Starting photon generation: " + particle)
            x_distances = e_x_distances if particle == "e-" else gamma_x_distances
            for x in x_distances:
                print("======> Distance: " + str(x))
                if counter == 0:
                    counts[particle][x] = []
                    scint_counts[particle][x] = []
                    cherenkov_counts[particle][x] = []
                position = (x * m, 0., 0.)
                # gen = g4gen.G4Generator(scint)
                g4 = G4Generator()		# Should not be necessary
                output = g4.generate(particle, position, momentum, scintillator, gen, energy=energy)

                track_tree = gen.track_tree
                if track_tree is not None:
                    count_processes.display_track_tree(gen.track_tree, particle)
                    track_count = len(track_tree) - 1
                    track_counts[particle][counter] = track_count

                    max_distance = 0.
                    center_location_track_number = 1 if particle == 'e-' else 2		# Note: this is tricky - be careful
                    center_location = track_tree[center_location_track_number]['position']
                    for key, entry in track_tree.iteritems():
                        if 'position' in entry:
                            position = entry['position']
                            '''   XX Looks like the type of 'position' may have change from an x,y,z to a (1,2,3) tuple
                            distance_from_center = math.sqrt(math.pow((position.x - center_location.x),2) +
                                math.pow((position.y - center_location.y),2) +
                                math.pow((position.z - center_location.z),2))
                            if distance_from_center > max_distance:
                            max_distance = distance_from_center
                            '''
                    vertex_positions[particle][counter] = max_distance

                counts[particle][x].append(len(output.pos))

                scint_count, cherenkov_count = count_processes.count_processes(output)
                scint_counts[particle][x].append(scint_count)
                cherenkov_counts[particle][x].append(cherenkov_count)
                if counts[particle][x][-1] != scint_counts[particle][x][-1] + cherenkov_counts[particle][x][-1]:
                    print("===>>> Uh oh: counts don't add up: ", particle, x, counts[particle][x], scint_counts[particle][x], cherenkov_counts[particle][x]);

                g4 = None
    print
    print("G4 state: ", Geant4.gStateManager.GetCurrentState())

def plot():
    # the histogram of the data
    # See: https://matplotlib.org/devdocs/gallery/pyplots/pyplot_text.html#sphx-glr-gallery-pyplots-pyplot-text-py
    # n, bins, patches = plt.hist(track_counts['e-']) # , 50, normed=1, facecolor='g', alpha=0.75)

    fig, axs = plt.subplots(nrows=2, ncols=2) # , sharey=True)
    fig.suptitle("Track Histograms | 100 particles each | 2 MeV", fontsize=14)
    # plt.title("Track counts")
    # plt.xlabel('Number of Tracks')

    plt1 = axs[0][0]
    plt1.hist(track_counts['e-'], bins=20, histtype='step', label='e$^-$', linewidth=2) # ,bins=400,histtype='step',linewidth=2,label='e$^-$')
    plt1.set_title('e -')
    plt1.grid(True)		# See: https://matplotlib.org/examples/pylab_examples/axes_props.html
    plt1.set_xlabel('Number of Tracks')

    plt2 = axs[0][1]
    plt2.hist(track_counts['gamma'], bins=20, histtype='step', label='gamma', linewidth=2) # ,bins=400,histtype='step',linewidth=2,label='e$^-$')
    plt2.set_title('gamma')
    plt2.grid(True)		# See: https://matplotlib.org/examples/pylab_examples/axes_props.html
    plt2.set_xlabel('Number of Tracks')

    plt3 = axs[1][0]
    plt3.hist(vertex_positions['e-'], bins=40, histtype='step', label='e$^-$', linewidth=2) # ,bins=400,histtype='step',linewidth=2,label='e$^-$')
    # plt3.set_title('e -')
    plt3.grid(True)		# See: https://matplotlib.org/examples/pylab_examples/axes_props.html
    plt3.set_xlabel('Max vertex distance from origin')

    plt4 = axs[1][1]
    plt4.hist(vertex_positions['gamma'], bins=40, histtype='step', label='gamma', linewidth=2) # ,bins=400,histtype='step',linewidth=2,label='e$^-$')
    # plt4.set_title('gamma')
    plt4.grid(True)		# See: https://matplotlib.org/examples/pylab_examples/axes_props.html
    plt4.set_xlabel('Max vertex distance from first deposition')

    # plt.legend()
    plt.show()
    exit(0)

    e_avgs, e_yerr = compute_stats(counts['e-'], e_x_distances)
    gamma_avgs, gamma_yerr = compute_stats(counts['gamma'], gamma_x_distances)
    e_scint_avgs, _ = compute_stats(scint_counts['e-'], e_x_distances)
    gamma_scint_avgs, _ = compute_stats(scint_counts['gamma'], gamma_x_distances)
    e_cheren_avgs, _ = compute_stats(cherenkov_counts['e-'], e_x_distances)
    gamma_cheren_avgs, _ = compute_stats(cherenkov_counts['gamma'], gamma_x_distances)

    # See: https://matplotlib.org/1.2.1/examples/pylab_examples/errorbar_demo.html
    fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True)
    fig.suptitle("Photon Counts | 2 MeV particles fired in +X direction | " + str(run_count) + " particles per position", fontsize=14)
    plt.title("Photon counts")
    plt.xlabel('X Distance')
    plt.ylabel('Photon Count')

    # See https://stackoverflow.com/questions/12957582/matplotlib-plot-yerr-xerr-as-shaded-region-rather-than-error-bars
    # for error shading
    plt1 = axs[0]
    plt1.set_title('e -')
    plt1.grid(True)		# See: https://matplotlib.org/examples/pylab_examples/axes_props.html
    plt1.stackplot(e_x_distances, e_scint_avgs, e_cheren_avgs)
    plt1.legend(["Scintillation","Cherenkov"])
    '''
    plt1.errorbar(e_x_distances, e_avgs, yerr=e_yerr, color='#CC4F1B', capsize=3)
    #plt1.plot(e_x_distances, e_y, color='#CC4F1B')
    plt1.fill_between(e_x_distances, e_avgs - e_yerr, e_avgs + e_yerr, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    '''

    plt2 = axs[1]
    plt2.set_title('gamma')
    plt2.grid(True)

    plt2.stackplot(gamma_x_distances, gamma_scint_avgs, gamma_cheren_avgs)
    plt2.legend(["Scintillation","Cherenkov"])

    '''
    #plt2.xlabel('X Distance')
    #plt2.ylabel('Photon Count')
    plt2.errorbar(gamma_x_distances, gamma_avgs, yerr=gamma_yerr, capsize=3)
    plt2.fill_between(gamma_x_distances, gamma_avgs - gamma_yerr, gamma_avgs + gamma_yerr, alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF')
    '''
    plt.show()


    # This is how to reset the seed between runs
    #Geant4.HepRandom.setTheSeed(9876)


    # I think we need to reset in between - one seems to affect the other???  And if reverse the order...
    # But cant just regenerate the detector: Geant4 kernel is not PreInit state : Method ignored.
    #	Then seg fault

    '''
    plt.hist(out_ph1.wavelengths,bins=400,histtype='step',linewidth=2,label='e$^-$')
    plt.hist(out_ph2.wavelengths,bins=400,histtype='step',linewidth=2,label='$\gamma$')
    plt.xlabel('scintillation photons wavelength [nm]')
    plt.legend()
    plt.show()
    plt.close()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(out_ph1.pos[:,0],out_ph1.pos[:,1],out_ph1.pos[:,2],'.',label='e$^-$')
    #ax.plot(out_ph2.pos[:,0],out_ph2.pos[:,1],out_ph2.pos[:,2],'.',label='$\gamma$')
    plt.legend()
    plt.show()
    '''

    # Trying to dump Geant4 state to start over - really dump Geant4 entirely and reload
    print("g4gen ref count: ", sys.getrefcount(g4gen))
    # gen = None
    '''
    print("g4gen ref count: ", sys.getrefcount(g4gen))
    del sys.modules["chroma.generator.g4gen"]		# This doesn't actually show in teh list - so why does this work???
    del sys.modules["chroma.generator.Geant4"]
    del sys.modules["chroma.generator.g4py"]
    del sys.modules["chroma.generator"]
    del g4gen
    '''

    #Geant4.gApplyUICommand("/run/abort")
    #Geant4.gApplyUICommand("/run/geometryModified")
    '''
    g4 = G4Generator()
    out_ph1 = g4.generate('e-', (0. * m, 0., 0.), momentum, scintillator, gen, energy=2.)		# NOTE the energy here.  for testing !!!!!
    gen.reset_g4()
    gen = None
    gen2 = g4gen.G4Generator(scintillator)
    out_ph1 = g4.generate('e-', (0. * m, 0., 0.), momentum, scintillator, gen2, energy=2.)		# NOTE the energy here.  for testing !!!!!
    '''

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

# See: https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def make_vector(start, end):
    return np.asarray(end) - np.asarray(start)

def compute_scatter_angle(track_tree, total_photons, particle_num, energy):
    neutron = None
    first_proton = None
    capture = None
    for key, value in track_tree.iteritems():
        if key == 1:
            neutron = value
        elif key == 2:
            first_proton = value
        elif ('process' in value and value['process'] == 'nCapture') and ('particle' in value and value['particle'] == 'gamma'):
            capture = value
    if first_proton is None:
        print('======= No first photon found ========')
        pprint.pprint(track_tree)
        print('=======')
    else:
        neutron_start_position  = neutron['position']
        first_proton_location   = first_proton['position']
        capture_location        = capture['position']
        initial_neutron_vector  = make_vector(neutron_start_position, first_proton_location)
        neutron_recoil_vector   = make_vector(first_proton_location, capture_location)
        energy_pct_in_first_photon = first_proton['child_processes']['Scintillation'] / (total_photons * 1.) if 'Scintillation' in first_proton['child_processes'] else 0

        print('Particle #:\t' + str(particle_num) + '\t' + str(energy) +
              '\t' + str(neutron_start_position) + '\t' + str(first_proton_location) + '\t' + str(capture_location) +
              '\t' + str(np.degrees(angle_between(initial_neutron_vector, neutron_recoil_vector))) +
              '\t' + '{:.1%}'.format(energy_pct_in_first_photon) +
              '\t' + str(total_photons)
        )
        '''
        print("========")
        pprint.pprint(neutron)
        pprint.pprint(first_proton)
        pprint.pprint(capture)
        print("========")
        '''


def fire_particles(particle, count, energy, position, momentum):
    print('Particle\t#\tEnergy\tLocations: neutron\tproton\tcapture\t' +
          'Recoil angle' + '\tFirst proton energy %' + '\tTotal photons')
    scintillator = lensmaterials.create_scintillation_material()

    g4_params = G4DetectorParameters(world_material='G4_AIR', orb_radius=10.)
    gen = g4gen.G4Generator(scintillator, g4_detector_parameters=g4_params) # , physics_list=QGSP_BERT_HP())
    # gen = g4gen.G4Generator(scint)
    g4 = G4Generator()  # Should not be necessary
    photon_counts = []
    title = str(energy) + ' MeV ' + particle
    for i in range(count):
        print('=== Firing ' + particle + ' # ' + str(i) + ' ===')
        output = g4.generate(particle, position, momentum, scintillator, gen, energy=energy)
        track_tree = gen.track_tree
        pprint.pprint(track_tree)
        #pprint.pprint(output.__dict__)
        photon_counts.append(len(output.dir))
        # print('Photon count: ' + str(len(output.dir)))
        # Geant4.HepRandom.setTheSeed(9876)
        # print("Random seed: ", Geant4.HepRandom.getTheSeed()
        #plot_vertices(track_tree, title, file_name='vertices'+'-'+particle+'-'+str(i)+'.pickle', with_electrons=False)
        compute_scatter_angle(track_tree, len(output.dir), i , energy)
    print(photon_counts)
    print(np.average(photon_counts))
    print(np.std(photon_counts))




if __name__ == '__main__':

    '''
    Geant4.gApplyUICommand("/run/verbose 2")
    Geant4.gApplyUICommand("/event/verbose 2")
    Geant4.gApplyUICommand("/tracking/verbose 2")
    '''

    ev_dict = os.environ
    pprint.pprint(ev_dict)

    print("g4gen ref count: ", sys.getrefcount(g4gen))

    print("G4 state: ", Geant4.gStateManager.GetCurrentState())
    print("Random engine: ", Geant4.HepRandom.getTheEngine())
    print("Random seed: ", Geant4.HepRandom.getTheSeed())

    ##### Debugging stuff pulled from other places #####
    # Various futzing around with world material
    # helium = G4Material.GetMaterial("G4_HE")
    # if helium == None:
    #    helium = gNistManager.FindOrBuildMaterial("G4_HE")  # Isotopes parameter?
    # If we didn't find it - squawk
    # self.world_material = helium
    # self.world_material = G4Material.GetMaterial("G4_AIR")
    # self.world_material = self.create_g4material(he)

    # Debugging stuff pulled out of g4gen.py
    element_table = Geant4.gElementTable        # Was: G4Element.GetElementTable() in c++
    # Somehow one of these blows stuff up...
    # element_table2 = gElementTable
    # material_table = gMaterialTable

    # print("create_g4material() number of elements: ", G4Element.GetNumberOfElements())

    import g4py.NISTmaterials

    # Add Geant4 PrintVersion()
    Geant4.print_version()     # Geant4 method
    #G4NistManager * NISTManager = G4NistManager::Instance();
    #NISTManager->ListMaterials("all");
    ##########

    # fire_particles(['mu-'], 1, 4.*100.)   # ['mu-','mu+'], 2, 4.*100.)   # mu+ is the anti-particle (but I think it has negative charge)
    # fire_particles(['mu-','mu+'], 2, 1.*1000.)   # mu+ is the anti-particle (but I think it has negative charge)
    #fire_particles(['e-','gamma'], 2, 2.)
    for energy in [2.]:  # ,20.,200.]:
        fire_particles('neutron', 1, energy, (0,0,0), (1,0,0))        # Can the momentum override the energy???
