import pickle
import matplotlib.pyplot as plt
import numpy as np

from DetectorResponseGaussAngle import DetectorResponseGaussAngle
from EventAnalyzer import EventAnalyzer
import nog4_sim

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
	rad = 1.0		# Location of event - will be DEPRECATED

	analyzer = EventAnalyzer(det_res)
	events, points = create_single_source_events(rad, sig_pos, amount, repetition)

	sig_cone = 0.01
	lens_dia = None
	n_ph = 0
	min_tracks = 0.1
	chiC = 1.5.
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
