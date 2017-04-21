import time,os,argparse
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
from mpl_toolkits.mplot3d import Axes3D


# To run this script: python first_test_jacopo.py 1 will plot the simulation of one track distance distribution from one source
# python first_test_jacopo.py 2 will run the simulation with two sources with various separation and relative amount
# add the flag -save to save the histogram without plotting it

def fixed_dist(sample,radius):
	loc1 = sph_scatter(sample,in_shell = 1000,out_shell = 2000)
	loc2 = sph_scatter(sample)
	rads = np.linspace(0.01*radius,0.1*radius,sample)
	dist = loc1-loc2
	loc2 = loc1 + np.einsum('i,ij->ij',rads,dist/np.linalg.norm(dist,axis=1)[:,None])
	bl_idx = np.linalg.norm(loc2,axis=1)>radius
	loc2[bl_idx] = 2 * loc1[bl_idx] - loc2[bl_idx]
	return loc1,loc2,rads

def sph_scatter(sample,in_shell = 0,out_shell = 5000):
	loc = np.random.uniform(-out_shell,out_shell,(sample,3))
	while len(loc[(np.linalg.norm(loc,axis=1)>in_shell) & (np.linalg.norm(loc,axis=1)<=out_shell)]) != sample:
		bl_idx = np.logical_not((np.linalg.norm(loc,axis=1)>in_shell) & (np.linalg.norm(loc,axis=1)<=out_shell))
		smpl = sum(bl_idx)
		loc[bl_idx] = np.random.uniform(-out_shell,out_shell,(smpl,3))
	return loc


def plot_sphere(loc1,loc2=None,rads=None):
	u = np.linspace(0, 2 * np.pi, 100)
	v = np.linspace(0, np.pi, 100)
	x = np.outer(np.cos(u), np.sin(v))
	y = np.outer(np.sin(u), np.sin(v))
	z = np.outer(np.ones(np.size(u)), np.cos(v))
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	if loc2 == None:
		ax.plot(loc1[:,0]/5000,loc1[:,1]/5000,loc1[:,2]/5000,'.')
	else:
		color = plt.cm.rainbow(np.linspace(0, 1, len(rads)))
		for i,j,lab,col in zip(loc1/5000,loc2/5000,rads,color):
			ax.plot([i[0],j[0]],[i[1],j[1]],[i[2],j[2]],'.',c=col,label='distance %0.0f mm'%lab)
		plt.legend(ncol=4, prop={'size': 10})
	ax.plot_surface(x,y,z,linewidth=0.2,alpha=0.1)
	plt.show()
	plt.close()

def create_double_source_events(locs1, locs2, sigma, amount1, amount2):
	# produces a list of Photons objects, each with two different photon sources
	# locs1 and locs2 are lists of locations (tuples)
	# other parameters are single numbers
	events = []
	if len(locs1.shape) == 1:
		locs1 = locs1.reshape((1,-1))
		locs2 = locs2.reshape((1,-1))
	for loc1,loc2 in zip(locs1,locs2):
	    event1 = kbl.gaussian_sphere(loc1, sigma, int(amount1))
	    event2 = kbl.gaussian_sphere(loc2, sigma, int(amount2))
	    event = event1 + event2						#Just add the list of photons from the two sources into a single event
	    events.append(event)
	return events

def remove_nan(dist,ofst_diff,drct):
	if np.isnan(dist).any():
		idx = np.where(np.isnan(dist))[0]
		dist[idx] = np.absolute(np.cross(ofst_diff[idx],drct[idx]))/np.linalg.norm(drct[idx],axis=1).reshape(-1,1)
	return dist

def syst_solve(drct,r_drct,ofst_diff):
	s_a = np.einsum('ij,ij->i',drct,drct)
	s_b = np.einsum('ij,ij->i',r_drct,r_drct)
	d_dot = np.einsum('ij,ij->i',drct,r_drct)
	q1 = np.einsum('ij,ij->i',-ofst_diff,drct)
	q2 = np.einsum('ij,ij->i',-ofst_diff,r_drct)
	matr = np.stack((np.vstack((s_a,-d_dot)).T,np.vstack((d_dot,-s_b)).T),axis=1)
	if any(np.linalg.det(matr)==0):
		matr[np.linalg.det(matr)==0] = np.identity(2)
	qt = np.vstack((q1,q2)).T
	return np.linalg.solve(matr,qt)

def track_dist(ofst,drct,sgm=None):
	half = ofst.shape[0]/2
	arr_dist = []
	arr_sgm = []
	for i in range(1,(ofst.shape[0]-1)/2+1):
		ofst_diff = ofst - np.roll(ofst,i,axis=0)
		r_drct = np.roll(drct,i,axis=0)
		r_sgm = np.roll(sgm,i,axis=0)
		sm = np.stack((sgm,r_sgm),axis=1)
		b_drct = np.cross(drct,r_drct)
		norm_d = b_drct/np.linalg.norm(b_drct,axis=1).reshape(-1,1)
		dist = np.absolute(np.einsum('ij,ij->i',ofst_diff,norm_d))
		remove_nan(dist,ofst_diff,drct)
		arr_dist.extend(dist)
		if sgm != None:
			multp = syst_solve(drct,r_drct,ofst_diff)
			multp[np.where(multp==0)] = 1
			arr_sgm.extend(np.linalg.norm(np.einsum('ij,ij->ij',sm,multp),axis=1))
	if ofst.shape[0] & 0x1: pass						#condition removed if degeneracy is kept
	else:
		ofst_diff = ofst[:half]-np.roll(ofst,half,axis=0)[:half]
		r_drct = np.roll(drct,half,axis=0)[:half]
		r_sgm = np.roll(sgm,half,axis=0)[:half]
		sm = np.stack((sgm[:half],r_sgm),axis=1)
		b_drct = np.cross(drct[:half],r_drct)
		norm_d = b_drct/np.linalg.norm(b_drct,axis=1).reshape(-1,1)
		dist = np.absolute(np.einsum('ij,ij->i',ofst_diff,norm_d))
		remove_nan(dist,ofst_diff,drct[:half])
		arr_dist.extend(dist)
		if sgm != None:
			multp = syst_solve(drct[:half],r_drct,ofst_diff)
			multp[np.where(multp==0)] = 1
			arr_sgm.extend(np.linalg.norm(np.einsum('ij,ij->ij',sm,multp),axis=1))
	if sgm != None:
		arr_sgm = np.asarray(arr_sgm)
		return arr_dist,arr_sgm
	else:
		return arr_dist

def plot_hist(arr_dist,out_print,save=None,directory=None):
	tot_tracks = len(arr_dist)
	plt.hist(np.asarray(arr_dist),bins=1000,normed=True)
	plt.xlim((0,7500))
	plt.ylim((1e-7,1e-2))
	plt.xlabel('reciprocal distance [mm]')
	plt.title('Distance distribution for %s with a total of %i entries'%(out_print,tot_tracks))
	plt.yscale('log')
	plt.subplots_adjust(left=0.03, right=0.99, top=0.97, bottom=0.06)
	if save == False:
		plt.show()
	else:
		F = pylab.gcf()
		ds = F.get_size_inches()
		F.set_size_inches((ds[0]*2.45,ds[1]*1.83))
		if directory == None:
			F.savefig(out_print.replace(' ','')+'.png')
		else:
			F.savefig(directory+out_print.replace(' ','')+'.png')

def create_event(location, sigma, amount, config,in_file,sgm=None):
	#simulates a single event within the detector for a given configuration adapted from kambamland2.
	fname = 'SS'
	kabamland = Detector(lm.ls)
	kbl.build_kabamland(kabamland, config)
	kabamland.flatten()
	kabamland.bvh = load_bvh(kabamland)
	sim = Simulation(kabamland)
	det_res = DetectorResponseGaussAngle(config,10,10,10,in_file)
	analyzer = EventAnalyzer(det_res)
	arr_dist = []
	arr_sgm = []
	if type(location) != np.ndarray:
		location = location.reshape((1,-1))
	for lg in location:
		start_time = time.time()
		sim_event = kbl.gaussian_sphere(lg, sigma, amount)
		l_arr_dist = []
		l_arr_sgm = []
		for ev in sim.simulate(sim_event, keep_photons_beg = True, keep_photons_end = True, run_daq=False, max_steps=100):
			tracks = analyzer.generate_tracks(ev)
			if sgm != None:
				tr_dist,err_dist = track_dist(tracks.hit_pos.T,tracks.means.T,sgm=tracks.sigmas)
				l_arr_sgm.extend(err_dist)
			else:
				tr_dist = track_dist(tracks.hit_pos.T,tracks.means.T)
			l_arr_dist.extend(tr_dist)
		arr_dist.append(l_arr_dist)
		arr_sgm.append(np.asarray(l_arr_sgm))
	if sgm != None:
		arr_sgm = np.asarray(arr_sgm)
		return arr_dist,1./arr_sgm
	else:
		return arr_dist


def double_event_eff_test(config, detres=None, detbins=10, n_repeat=10, sig_pos=0.01, n_ph_sim=300, n_ratio=10, n_pos=10, max_rad_frac=1.0, loc1=(0,0,0), max_int=0.99,sgm=None):
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
	start_time = time.time()
	ratios = np.linspace(0.5,max_int,n_ratio)
	amount1 = n_ph_sim*ratios
	amount2 = n_ph_sim*(1.0-ratios)
	if loc1 == None: 
		locs1, locs2, rads = fixed_dist(5,5000)
		plot_sphere(locs1,loc2=locs2,rads=rads)
	else:
		max_rad = min(max_rad_frac*det_res.inscribed_radius, det_res.inscribed_radius-np.linalg.norm(np.array(loc1)))
		rads = np.linspace(max_rad/n_pos,max_rad,n_pos)
		locs1 = np.tile(np.asarray([loc1]*n_repeat),(len(rads),1,1))
		locs2 = np.multiply(np.tile(uniform_sphere(n_repeat),(len(rads),1,1)),rads.reshape(len(rads),1,1))+loc1
	tot_arr_dist = []
	tot_arr_sgm = []
	for a1,a2 in zip(amount1,amount2):
		directory = 'ratio_%0.2f/' %(a1/n_ph_sim)
		if not os.path.exists(directory):
    			os.makedirs(directory)
		for lc1,lc2 in zip(locs1,locs2):
			out_print = "ratio %0.2f - separation %d" %(a1/n_ph_sim,np.linalg.norm(lc1-lc2))
			print out_print
			sim_events = create_double_source_events(lc1, lc2, sig_pos, a1, a2)
			arr_dist = []
			arr_sgm = []
			for ev in sim.simulate(sim_events, keep_photons_beg = True, keep_photons_end = True, run_daq=False, max_steps=100):
				tracks = analyzer.generate_tracks(ev)
				if sgm != None:
					tr_dist,err_dist = track_dist(tracks.hit_pos.T,tracks.means.T,sgm=tracks.sigmas)
					arr_sgm.extend(err_dist)
				else:
					tr_dist = track_dist(tracks.hit_pos.T,tracks.means.T)
				arr_dist.extend(tr_dist)			
			tot_arr_dist.append(arr_dist)
			tot_arr_sgm.append(np.asarray(arr_sgm))
	if sgm != None:
		return tot_arr_dist, 1./np.asarray(tot_arr_sgm), rads
	else:
		return tot_arr_dist, rads
