import time,os
import numpy as np
import lensmaterials as lm
import kabamland2 as kbl
import matplotlib.pyplot as plt
import pylab
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
	    event1 = kbl.gaussian_sphere(loc1, sigma, int(amount1))
	    event2 = kbl.gaussian_sphere(loc2, sigma, int(amount2))
	    event = event1 + event2 # Just add the list of photons from the two sources into a single event
	    events.append(event)
	return events

def remove_nan(dist,ofst_diff,drct):
	if np.isnan(dist).any():
		idx = np.where(np.isnan(dist))[0]
		dist[idx] = np.absolute(np.cross(ofst_diff[idx],drct[idx]))/np.linalg.norm(drct[idx],axis=1).reshape(-1,1)
	return dist


def track_dist(ofst,drct):
	half = ofst.shape[0]/2
	arr_dist = []
	for i in range(1,(ofst.shape[0]-1)/2+1):	# ofst.shape[0] for keeping the degeneracy
		ofst_diff = ofst-np.roll(ofst,i,axis=0)
		b_drct = np.cross(drct,np.roll(drct,i,axis=0))
		norm_d = b_drct/np.linalg.norm(b_drct,axis=1).reshape(-1,1)
		dist = np.absolute(np.einsum('ij,ij->i',ofst_diff,norm_d))
		remove_nan(dist,ofst_diff,drct)
		arr_dist.extend(dist)
	if ofst.shape[0] & 0x1: pass			#condition removed if degeneracy is kept
	else:
		ofst_diff = ofst[:half]-np.roll(ofst,half,axis=0)[:half]
		b_drct = np.cross(drct,np.roll(drct,half,axis=0))[:half]
		norm_d = b_drct/np.linalg.norm(b_drct,axis=1).reshape(-1,1)
		dist = np.absolute(np.einsum('ij,ij->i',ofst_diff,norm_d))
		remove_nan(dist,ofst_diff,drct)
		arr_dist.extend(dist)
	return arr_dist

def plot_hist(arr_dist,out_print,tot_tracks,directory=None):
	plt.hist(np.asarray(arr_dist),bins=1000,normed=True)
	plt.xlim((0,7500))
	plt.ylim((1e-7,1e-2))
	plt.xlabel('reciprocal distance [mm]')
	plt.title('Distance distribution for %s with a total of %i entries'%(out_print,tot_tracks))
	plt.yscale('log')
	plt.subplots_adjust(left=0.03, right=0.99, top=0.97, bottom=0.06)
	F = pylab.gcf()
	ds = F.get_size_inches()
	F.set_size_inches((ds[0]*2.45,ds[1]*1.83))
	if directory == None: F.savefig(out_print.replace(' ','')+'.png')
	else: F.savefig(directory+out_print.replace(' ','')+'.png')
	plt.close()


def create_event(location, sigma, amount, config,in_file):
	#simulates a single event within the detector for a given configuration adapted from kambamland2.
	fname = 'SS'
	kabamland = Detector(lm.ls)
	kbl.build_kabamland(kabamland, config)
	kabamland.flatten()
	kabamland.bvh = load_bvh(kabamland)
	sim = Simulation(kabamland)
	det_res = DetectorResponseGaussAngle(config,10,10,10,in_file)
	analyzer = EventAnalyzer(det_res)
	sim_event = kbl.gaussian_sphere(location, sigma, amount)
	arr_dist = []
	for ev in sim.simulate(sim_event, keep_photons_beg = True, keep_photons_end = True, run_daq=False, max_steps=100):
		tracks = analyzer.generate_tracks(ev)
		arr_dist.extend(track_dist(tracks.hit_pos.T,tracks.means.T))
	plot_hist(arr_dist,fname,len(arr_dist))


def double_event_eff_test(config, detres=None, detbins=10, n_repeat=10, sig_pos=0.01, n_ph_sim=300, n_ratio=10, n_pos=10, max_rad_frac=1.0, loc1=(0,0,0)):
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
	sim = Simulation(kabamland)
	print "Simulation started."

	det_res = DetectorResponseGaussAngle(config, detbins, detbins, detbins, detres)
	analyzer = EventAnalyzer(det_res)
	
	# make list of radii and energy ratios
	# Keep events inside detector
	max_rad = min(max_rad_frac*det_res.inscribed_radius, det_res.inscribed_radius-np.linalg.norm(np.array(loc1)))
	start_time = time.time()
	rads = np.linspace(max_rad/n_pos,max_rad,n_pos)
	ratios = np.linspace(0.5,0.99,n_ratio)
	amount1 = n_ph_sim*ratios
	amount2 = n_ph_sim*(1.0-ratios)
	locs1 = np.asarray([loc1]*n_repeat)
	locs2 = np.multiply(np.tile(uniform_sphere(n_repeat),(len(rads),1,1)),rads.reshape(len(rads),1,1))+loc1

	for a1,a2 in zip(amount1,amount2):
		directory = 'ratio_%0.2f/' %(a1/n_ph_sim)
		if not os.path.exists(directory):
    			os.makedirs(directory)
		for loc2 in locs2:
			out_print = "ratio %0.2f - separation %d" %(a1/n_ph_sim,np.linalg.norm(loc2[0]))
			print out_print
			sim_events = create_double_source_events(locs1, loc2, sig_pos, a1, a2)
			arr_dist = []
			for ev in sim.simulate(sim_events, keep_photons_beg = True, keep_photons_end = True, run_daq=False, max_steps=100):
				tracks = analyzer.generate_tracks(ev)
				arr_dist.extend(track_dist(tracks.hit_pos.T,tracks.means.T))
			plot_hist(arr_dist,out_print,len(arr_dist),directory)
			

if __name__ == '__main__':
	datadir = "/home/miladmalek/TestData/"#"/home/skravitz/TestData/"#
	fileinfo = 'cfJiani3_4'#'configpc6-meniscus6-fl1_027-confined'#'configpc7-meniscus6-fl1_485-confined'#'configview-meniscus6-fl2_113-confined'
	in_file = datadir+'detresang-'+fileinfo+'_noreflect_100million.root'
	create_event((0,0,0), 0.01,4000,fileinfo,in_file)
	#double_event_eff_test(fileinfo,detres=in_file, detbins=10, n_repeat=10, sig_pos=0.01, n_ph_sim=4000, n_ratio=10, n_pos=5, max_rad_frac=0.2, loc1=(0,0,0))
