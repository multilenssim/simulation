from DetectorResponseGaussAngle import DetectorResponseGaussAngle
from EventAnalyzer import EventAnalyzer
from chroma.detector import Detector, G4DetectorParameters
from chroma.loader import load_bvh
from chroma.sim import Simulation
import time, h5py, os, argparse
import lensmaterials as lm
import kabamland2 as kbl
import numpy as np

import paths

def fixed_dist(sample,radius,rads=None):
	loc1 = sph_scatter(sample,in_shell = 4000,out_shell = 5000)
	loc2 = sph_scatter(sample,in_shell = 4000,out_shell = 5000)
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
	#kabamland = Detector(lm.get_scintillation_material())
	#kabamland.orb_radius = 4.5
	#kbl.build_kabamland(kabamland, config)
	#kabamland.flatten()
	#kabamland.bvh = load_bvh(kabamland)
        g4_detector_parameters=G4DetectorParameters(orb_radius=7., world_material='G4_Galactic')
	kabamland = kbl.load_or_build_detector(config, lm.create_scintillation_material(), g4_detector_parameters=g4_detector_parameters)
	sim = Simulation(kabamland,geant4_processes=1)
	det_res = DetectorResponseGaussAngle(config,10,10,10,in_file)
	analyzer = EventAnalyzer(det_res)
	return sim, analyzer

def run_simulation_and_write_events_file(file, sim, events, analyzer, first=False):
	arr = []
	for ev in sim.simulate(events, keep_photons_beg = True, keep_photons_end = True, run_daq=False, max_steps=100):
		tracks = analyzer.generate_tracks(ev)
		#pprint(vars(ev))
		print('Firing particle name/photon count/track count/location/direction: \t' +  # Add energy
		      ev.primary_vertex.particle_name + '\t' +
		      str(len(ev.photons_beg)) + '\t' +
		      str(len(tracks)) + '\t' +
		      str(ev.primary_vertex.pos) + '\t' +
		      str(ev.primary_vertex.dir) + '\t')
		print('Photons begin count, track count:\t' + str(len(ev.photons_beg)) + '\t' + str(len(tracks)))
		if first:
			coord = file.create_dataset('coord',data=[tracks.hit_pos.T, tracks.means.T],chunks=True)
			uncert = file.create_dataset('sigma',data=tracks.sigmas,chunks=True)
			arr.append(tracks.sigmas.shape[0])
			file.create_dataset('r_lens',data=tracks.lens_rad)
		else:
			coord = file['coord']
			uncert = file['sigma']
			coord.resize(coord.shape[1]+tracks.means.shape[1], axis=1)
			coord[:,-tracks.means.shape[1]:,:] = [tracks.hit_pos.T, tracks.means.T]
			uncert.resize(uncert.shape[0]+tracks.sigmas.shape[0], axis=0)
			uncert[-tracks.sigmas.shape[0]:] = tracks.sigmas
			arr.append(uncert.shape[0])
	return arr

def fixed_dist_hist(dist,sample,amount,sim,analyzer,sigma=0.01):
	arr = []
	first = True
	locs1, locs2, rad = fixed_dist(sample,5000,rads=dist)
	fname = 'd-site'+str(int(dist/10))+'cm.h5'
	print('File: ' + path + fname)
	with h5py.File(path+fname,'w') as f:
		for lc1,lc2 in zip(locs1,locs2):
			sim_events = create_double_source_events(lc1, lc2, sigma, amount/2, amount/2)
			arr.append(run_simulation_and_write_events_file(f, sim, sim_events, analyzer, first))
			first = False
		f.create_dataset('idx',data=arr)


def bkg_dist_hist(sample,amount,sim,analyzer,sigma=0.01):
	arr = []
	first = True
	i = 0
	location = sph_scatter(sample,in_shell = 4000,out_shell = 5000)
	fname = 's-site.h5'
	print('File: ' + path + fname)
	with h5py.File(path+fname,'w') as f:
		for lg in location:
			sim_event = kbl.gaussian_sphere(lg, sigma, amount)
			arr.append(run_simulation_and_write_events_file(f, sim, sim_event, analyzer, first))
			first = False
		f.create_dataset('idx',data=arr)

from chroma.sample import uniform_sphere

def myhack():
    while True:
        yield uniform_sphere()
        #yield [-1,0,0]

def fire_particles(particle_name,sample,energy,sim,analyzer,sigma=0.01):
	arr = []
	first = True
	location = sph_scatter(sample)
	fname = particle_name+'.h5'
	with h5py.File(path+fname,'w') as f:
		for lg in location:     # x in np.linspace(0., 1000., num=20):
			#lg = [7000.,0,0]
			# direction = [-1,0,0]
			# Direction original code is: vertex.isotropic()
			gun = vertex.particle_gun([particle_name], vertex.constant(lg), vertex.isotropic(), vertex.flat(float(energy) * 0.99, float(energy) * 1.01))
			# gun = vertex.particle_gun([particle_name], vertex.constant(lg), myhack(), vertex.flat(float(energy) * 0.99, float(energy) * 1.01))
			arr.append(run_simulation_and_write_events_file(f, sim, gun, analyzer, first))
			first = False
		f.create_dataset('idx',data=arr)

data_file_prefix = '/home/kwells/ch_hdf5_files/'

energy = 1.

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('cfg', help='detector configuration')
	args = parser.parse_args()
	sample = 1000
	distance = np.linspace(100,700,6)
	cfg = args.cfg
	seed_loc = 'r0-1'
	ptf = paths.get_data_file_path()
	path = ptf+seed_loc
	start_time = time.time()
	sim,analyzer = sim_setup(cfg,paths.get_calibration_file_name(cfg))
	print 'configuration loaded in %0.2f' %(time.time()-start_time)

	print('Firing ' + str(energy) + ' MeV e-''s')
	fire_particles('e-', sample, energy*MeV, sim, analyzer)
	print('Firing ' + str(energy) + ' MeV gammas')
	fire_particles('gamma', sample, energy*MeV, sim, analyzer)

	'''
	bkg_dist_hist(sample,13200,sim,analyzer)
	print 's-site done'
	for dst in distance:
		fixed_dist_hist(dst,sample,13200,sim,analyzer)
		print 'distance '+str(int(dst/10))+' done'
	'''
	print time.time()-start_time

