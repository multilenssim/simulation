from DetectorResponseGaussAngle import DetectorResponseGaussAngle
from EventAnalyzer import EventAnalyzer
from chroma.detector import Detector
from chroma.sim import Simulation
from chroma.loader import load_bvh
import lensmaterials as lm
import kabamland2 as kbl
import detectorconfig
import numpy as np
import time, h5py

def fixed_dist(sample,radius,rads=None):
	loc1 = sph_scatter(sample)
	loc2 = sph_scatter(sample)
	if rads == None:
		rads = np.linspace(50,500,sample)
	else:
		rads = np.full(sample,rads)
	dist = loc1-loc2
	loc2 = loc1 + np.einsum('i,ij->ij',rads,dist/np.linalg.norm(dist,axis=1)[:,None])
	bl_idx = np.linalg.norm(loc2,axis=1)>radius
	loc2[bl_idx] = 2 * loc1[bl_idx] - loc2[bl_idx]
	return loc1,loc2,rads

def sph_scatter(sample,in_shell = 0,out_shell = 1000):
	loc = np.random.uniform(-out_shell,out_shell,(sample,3))
	while len(loc[(np.linalg.norm(loc,axis=1)>in_shell) & (np.linalg.norm(loc,axis=1)<=out_shell)]) != sample:
		bl_idx = np.logical_not((np.linalg.norm(loc,axis=1)>in_shell) & (np.linalg.norm(loc,axis=1)<=out_shell))
		smpl = sum(bl_idx)
		loc[bl_idx] = np.random.uniform(-out_shell,out_shell,(smpl,3))
	return loc

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

def sim_setup(config,in_file):
	kabamland = Detector(lm.ls)
	kbl.build_kabamland(kabamland, config)
	kabamland.flatten()
	kabamland.bvh = load_bvh(kabamland)
	sim = Simulation(kabamland)
	det_res = DetectorResponseGaussAngle(config,10,10,10,in_file)
	analyzer = EventAnalyzer(det_res)
	return sim, analyzer

def fixed_dist_hist(dist,sample,amount,sim,analyzer,sigma=0.01):
	arr = []
	i = 0
	locs1, locs2, rad = fixed_dist(sample,5000,rads=dist)
	fname = 'd-site'+str(int(dist/10))+'cm.h5'
	with h5py.File(path+fname,'w') as f:
		for lc1,lc2 in zip(locs1,locs2):
			sim_events = create_double_source_events(lc1, lc2, sigma, amount/2, amount/2)
			for ev in sim.simulate(sim_events, keep_photons_beg = True, keep_photons_end = True, run_daq=False, max_steps=100):
				tracks = analyzer.generate_tracks(ev)
				if i == 0:
					coord = f.create_dataset('coord',data=[tracks.hit_pos.T, tracks.means.T],chunks=True)
					uncert = f.create_dataset('sigma',data=tracks.sigmas,chunks=True)
					arr.append(tracks.sigmas.shape[0])
					f.create_dataset('r_lens',data=tracks.lens_rad)
				else:
					coord.resize(coord.shape[1]+tracks.means.shape[1], axis=1)
					coord[:,-tracks.means.shape[1]:,:] = [tracks.hit_pos.T, tracks.means.T]
					uncert.resize(uncert.shape[0]+tracks.sigmas.shape[0], axis=0)
					uncert[-tracks.sigmas.shape[0]:] = tracks.sigmas
					arr.append(uncert.shape[0])
			i =+ 1
		f.create_dataset('idx',data=arr)


def bkg_dist_hist(sample,amount,sim,analyzer,sigma=0.01):
	arr = []
	i = 0
	location = sph_scatter(sample)
	fname = 's-site.h5'
	with h5py.File(path+fname,'w') as f:
		for lg in location:
			sim_event = kbl.gaussian_sphere(lg, sigma, amount)
			for ev in sim.simulate(sim_event, keep_photons_beg = True, keep_photons_end = True, run_daq=False, max_steps=100):
				tracks = analyzer.generate_tracks(ev)
				if i == 0:
					coord = f.create_dataset('coord',data=[tracks.hit_pos.T, tracks.means.T],chunks=True)
					uncert = f.create_dataset('sigma',data=tracks.sigmas,chunks=True)
					arr.append(tracks.sigmas.shape[0])
					f.create_dataset('r_lens',data=tracks.lens_rad)
				else:
					coord.resize(coord.shape[1]+tracks.means.shape[1], axis=1)
					coord[:,-tracks.means.shape[1]:,:] = [tracks.hit_pos.T, tracks.means.T]
					uncert.resize(uncert.shape[0]+tracks.sigmas.shape[0], axis=0)
					uncert[-tracks.sigmas.shape[0]:] = tracks.sigmas
					arr.append(uncert.shape[0])
			i =+ 1
		f.create_dataset('idx',data=arr)

	


if __name__ == '__main__':
	sample = 1000
	distance = np.linspace(100,700,6)
	cfg = 'cfJiani3_2'
	seed_loc = 'r0-1'
	path = '/home/jacopodalmasson/Desktop/dev/'+cfg+'/raw_data/'+seed_loc
	start_time = time.time()
	sim,analyzer = sim_setup(cfg,'/home/miladmalek/TestData/detresang-cfJiani3_2_1DVariance_100million.root')
	print 'configuration loaded in %0.2f' %(time.time()-start_time)
	bkg_dist_hist(sample,6600,sim,analyzer)
	print 's-site done'
	for dst in distance:
		fixed_dist_hist(dst,sample,6600,sim,analyzer)
		print 'distance '+str(int(dst/10))+' done'
	print time.time()-start_time
	
#'cfSam1_1'
#'/home/miladmalek/TestData/detresang-cfSam1_1_1DVariance_100million.root'
#'cfJiani3_4'
#/home/miladmalek/TestData/detresang-cfJiani3_4_1DVariance_100million.root
