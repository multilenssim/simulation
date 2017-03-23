import time
import numpy as np
import lensmaterials as lm
import kabamland2 as kbl
from EventAnalyzer import EventAnalyzer
from chroma.detector import Detector
from chroma.sim import Simulation
from chroma.sample import uniform_sphere
from chroma.loader import load_bvh
from DetectorResponseGaussAngle import DetectorResponseGaussAngle


def create_double_source_events(locs1, locs2, sigma, amount1, amount2):
	# produces a list of Photons objects, each with two different photon sources
	# locs1 and locs2 are lists of locations (tuples)
	# other parameters are single numbers
	events = []
	n_loc = min(len(locs1),len(locs2))
	for ind in range(n_loc):
	    loc1 = locs1[ind]
	    loc2 = locs2[ind]
	    event1 = kbl.gaussian_sphere(loc1, sigma, amount1)
	    event2 = kbl.gaussian_sphere(loc2, sigma, amount2)
	    event = event1 + event2 # Just add the list of photons from the two sources into a single event
	    events.append(event)
	return events




def double_event_eff_test(config, detres=None, detbins=10, n_repeat=10, sig_pos=0.01, n_ph_sim=300, n_ratio=10, n_pos=10, max_rad_frac=1.0, loc1=(0,0,0)):
	# Produces a plot of reconstruction efficiency for double source events (# of events w/ 
	# 2 reconstructed # vtcs/# of events) as a function of event separation and ratio of source photons.
	# Creates a simulation of the given config, etc. (actual lenses set in kabamland2.py)
	# Simulates events with total number of photons given by n_ph_sim, split into two sources
	# One source is set at loc1, while the other is varied in radial distance from loc1 
	# (up to max_rad_frac*inscribed radius in n_pos steps)
	# The size of each photon source is given by sig_pos
	# The ratio of photons from each source is also varied from 0.5 to 0.99 (n_ratio steps)
	# Each (pos, ratio) pair is repeated n_repeat times
 
	kabamland = Detector(lm.ls)
	kbl.build_kabamland(kabamland, config)
	kabamland.flatten()
	kabamland.bvh = load_bvh(kabamland)
	#view(kabamland)
	#quit()

	sim = Simulation(kabamland)
	print "Simulation started."

	if detres is None:
		det_res = DetectorResponseGaussAngle(config, detbins, detbins, detbins)
	else:
		det_res = DetectorResponseGaussAngle(config, detbins, detbins, detbins, infile=(datadir+detres))
	analyzer = EventAnalyzer(det_res)

	# make list of radii and energy ratios
	# Keep events inside detector
	max_rad = min(max_rad_frac*det_res.inscribed_radius, det_res.inscribed_radius-np.linalg.norm(np.array(loc1)))
	rads = [max_rad*float(ii+1)/n_pos for ii in range(n_pos)]
	ratios = [0.5+0.49*float(ii)/max(n_ratio-1,1) for ii in range(n_ratio)]
	rad_plot = np.tile(rads/det_res.inscribed_radius, (n_ratio,1))
	ratio_plot = np.tile(ratios, (n_pos,1)).T

	effs = np.zeros((n_ratio, n_pos))
	avg_errs = np.zeros((n_ratio, n_pos))
	#avg_disps = np.zeros((n_ratio, n_pos))
	for ix, ratio in enumerate(ratios):
		amount1 = n_ph_sim*ratio
		amount2 = n_ph_sim*(1.0-ratio)
		for iy, rad in enumerate(rads):
			print "Ratio: " + str(ratio) + ", radius: " + str(rad)
			locs1 = [loc1]*n_repeat # list
			locs2 = uniform_sphere(n_repeat)*rad+loc1 #np array
			sim_events = create_double_source_events(locs1, locs2, sig_pos, amount1, amount2)
			for ind, ev in enumerate(sim.simulate(sim_events, keep_photons_beg = True, keep_photons_end = True, run_daq=False, max_steps=100)):
				vtcs = analyzer.generate_tracks(ev)
				print vtcs.means[0]

if __name__ == '__main__':
	datadir = "/home/miladmalek/TestData/"#"/home/skravitz/TestData/"#
	fileinfo = 'cfJiani3_4'#'configpc6-meniscus6-fl1_027-confined'#'configpc7-meniscus6-fl1_485-confined'#'configview-meniscus6-fl2_113-confined'
	double_event_eff_test(fileinfo,detres='detresang-'+fileinfo+'_noreflect_100million.root', detbins=10, n_repeat=10, sig_pos=0.01, n_ph_sim=4000, n_ratio=10, n_pos=5, max_rad_frac=0.2, loc1=(0,0,0))

