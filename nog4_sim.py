import time, h5py, argparse
import kabamland2 as kbl

import numpy as np

from Geant4.hepunit import *

import paths
import utilities
import detectorconfig

def fixed_dist(sample, radius, in_shell, out_shell, rads=None):
	loc1 = utilities.sph_scatter(sample,in_shell,out_shell)
	loc2 = utilities.sph_scatter(sample,in_shell,out_shell)
	if np.array_equal(rads,None):
		rads = np.linspace(50,500,sample)
	else:
		rads = np.full(sample,rads)
	dist = loc1-loc2
	loc2 = loc1 + np.einsum('i,ij->ij',rads,dist/np.linalg.norm(dist,axis=1)[:,None])
	bl_idx = np.linalg.norm(loc2,axis=1)>radius
	loc2[bl_idx] = 2 * loc1[bl_idx] - loc2[bl_idx]
	return loc1,loc2,rads

def create_double_source_events(locs1, locs2, sigma, amount1, amount2):
	# produces a list of Photons objects, each with two different photon sources
	# locs1 and locs2 are lists of locations (tuples)
	# other parameters are single numbers
	events = []
	if len(np.shape(locs1)) == 1:
		locs1 = np.reshape(locs1,(1,-1))
		locs2 = np.reshape(locs2,(1,-1))
	for loc1,loc2 in zip(locs1,locs2):
		event1 = kbl.gaussian_sphere(loc1, sigma, int(amount1))
		event2 = kbl.gaussian_sphere(loc2, sigma, int(amount2))
		event = event1 + event2						#Just add the list of photons from the two sources into a single event
		events.append(event)
	return events

# Runs the simulation and writes the HDF5 file (except the index)
def run_simulation(file, sim, events, analyzer, first=False):
# def fixed_dist_hist(dist,sample,amount,sim,analyzer,sigma=0.01):  # Where did this come from?
	arr = []
	i = 0
	locs1, locs2, rad = fixed_dist(sample,5000,rads=dist)
	fname = 'd-site'+str(int(dist/10))+'cm.h5'
	with h5py.File(path+fname,'w') as f:
		for lc1,lc2 in zip(locs1,locs2):
			sim_events = create_double_source_events(lc1, lc2, sigma, amount/2, amount/2)
			for ev in sim.simulate(sim_events, keep_photons_beg = True, keep_photons_end = True, run_daq=False, max_steps=100):
				tracks = analyzer.generate_tracks(ev,qe=(1./3.))
				if i == 0:
					coord = f.create_dataset('coord', maxshape=(2,None,3), data=[tracks.hit_pos.T, tracks.means.T],chunks=True)
					uncert = f.create_dataset('sigma', maxshape=(None,), data=tracks.sigmas,chunks=True)
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

def myhack():
    while True:
        yield uniform_sphere()
        #yield [-1,0,0]

# Generate photons from a set of random locations within a spherical shell
def fire(sample,energy,sim,analyzer,path,amount,sigma=0.01):
	arr = []
	first = True
	# KW? fname = particle_name+'.h5'
	location = utilities.sph_scatter(sample,in_shell,out_shell)
	fname = 's-site.h5'
        file_name = path+fname
        print('Writing h5 file: ' + file_name)
	with h5py.File(file_name,'w') as f:
		for lg in location:
			sim_event = kbl.gaussian_sphere(lg, sigma, amount)
			for ev in sim.simulate(sim_event, keep_photons_beg = True, keep_photons_end = True, run_daq=False, max_steps=100):
				tracks = analyzer.generate_tracks(ev,qe=(1./3.))
				if first:
					coord = f.create_dataset('coord', maxshape=(2,None,3), data=[tracks.hit_pos.T, tracks.means.T],chunks=True)
					uncert = f.create_dataset('sigma', maxshape=(None,), data=tracks.sigmas,chunks=True)
					arr.append(tracks.sigmas.shape[0])
					f.create_dataset('r_lens',data=tracks.lens_rad)
				else:
					coord.resize(coord.shape[1]+tracks.means.shape[1], axis=1)
					coord[:,-tracks.means.shape[1]:,:] = [tracks.hit_pos.T, tracks.means.T]
					uncert.resize(uncert.shape[0]+tracks.sigmas.shape[0], axis=0)
					uncert[-tracks.sigmas.shape[0]:] = tracks.sigmas
					arr.append(uncert.shape[0])
			first = False
		f.create_dataset('idx',data=arr)

energy = 1.

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('config_name', help='detector configuration', nargs='?', default='cfSam1_K200_10')
	parser.add_argument('sl', help='seed_location')
	args = parser.parse_args()
	config_name = args.config_name
	seed_loc = args.sl

	sample = 500
	distance = np.linspace(20,450,6)
	in_shell = int(seed_loc[0])*1000
	out_shell = int(seed_loc[1])*1000
	print('Seed locations: ' + str(in_shell) + ' ' + str(out_shell))

	config = detectorconfig.get_detector_config(config_name)

	data_file_dir = paths.get_data_file_path(config_name)
	start_time = time.time()
	sim,analyzer = utilities.sim_setup(config,paths.get_calibration_file_name(config_name))
	print 'configuration loaded in %0.2f' %(time.time()-start_time)
        path=paths.get_data_file_path(config_name)
        amount = 1000 # 16000

	'''
	fire_photons_single_site(sample, amount, sim, analyzer, in_shell, out_shell)
	print 's-site done'
	for dst in distance:
		fire_photons_double_site(sample,16000,sim,analyzer, in_shell, out_shell, dst)
		print 'distance '+str(int(dst/10))+' done'
        '''

	print('Firing ' + str(energy) + ' MeV e-''s (not really - WTF)')
	fire(sample, energy*MeV, sim, analyzer, path, amount)
        '''
	print('Firing ' + str(energy) + ' MeV gammas (not really - WTF)')
	fire(sample, energy*MeV, sim, analyzer, path, amount)
        '''

	'''
	bkg_dist_hist(sample,16000,sim,analyzer)
	print 's-site done'
	for dst in distance:
		fixed_dist_hist(data_path,dst,sample,16000,sim,analyzer)
		print 'distance '+str(int(dst/10))+' done'
	'''
	print time.time()-start_time

