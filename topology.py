from mpl_toolkits.mplot3d import Axes3D
#from chroma.generator import vertex
import matplotlib.pyplot as plt
import h5py,time,argparse
#import nog4_sim as setup
import numpy as np
import paths
from logger_lfd import logger

import gc
import multiprocessing
from multiprocessing import Pool

def sim_ev(cfg,particle,lg,energy):
	sim,analyzer = setup.sim_setup(cfg, paths.get_calibration_file_name(cfg))
	print 'Configuration loaded'
	gun = vertex.particle_gun([particle], vertex.constant(lg), vertex.isotropic(), vertex.flat(energy*0.999, energy*1.001))
	for ev in sim.simulate(gun,keep_photons_beg=True, keep_photons_end=True, run_daq=False, max_steps=100):
		vert = ev.photons_beg.pos
		tracks = analyzer.generate_tracks(ev,qe=(1./3.))
	return vert, tracks

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

def roll_funct(ofst,drct,sgm,i,half=False):
	pt = []
	r_ofst = np.roll(ofst,i,axis=0)
	r_drct = np.roll(drct,i,axis=0)
	r_sgm = np.roll(sgm,i,axis=0)
	if half:
		ofst = ofst[:i]
		drct = drct[:i]
		sgm = sgm[:i]
		r_ofst = r_ofst[:i]
		r_drct = r_drct[:i]
		r_sgm = r_sgm[:i]		

	ofst_diff = ofst - r_ofst
	#b_drct = np.cross(drct,r_drct)
	#norm_d = b_drct/np.linalg.norm(b_drct,axis=1).reshape(-1,1)
	#dist = np.absolute(np.einsum('ij,ij->i',ofst_diff,norm_d))
	#dist = remove_nan(dist,ofst_diff,drct)
	sm = np.stack((sgm,r_sgm),axis=1)
	off_stack = np.stack((ofst,r_ofst),axis=1)
	drct_stack = np.stack((drct,r_drct),axis=1)
	multp = syst_solve(drct,r_drct,ofst_diff)
	#multp[multp==0] = 1
	sigmas = np.linalg.norm(np.einsum('ij,ij->ij',sm,multp),axis=1)
	poc = off_stack + np.einsum('ijk,ij->ijk',drct_stack,multp)
	dist = poc[:,0,:] - poc[:,1,:]
	gc.collect()
	return np.linalg.norm(dist,axis=1),sigmas,np.mean(poc,axis=1)

def track_dist(ofst,drct,sgm,dim_len=0):
	half = ofst.shape[0]/2
	arr_dist, arr_sgm, arr_pos = [], [], []
	count = (ofst.shape[0]-1)/2+1
	for i in range(1,(ofst.shape[0]-1)/2+1):
		dist,sigmas,recon_pos = roll_funct(ofst,drct,sgm,i,half=False)
		arr_dist.extend(dist)
		arr_sgm.extend(sigmas)
		arr_pos.extend(recon_pos)
		logger.info('Count ' + str(i) + ' of ' + str(count))
	if ofst.shape[0] & 0x1:
		pass						#condition removed if degeneracy is kept
	else:
		dist,sigmas,recon_pos = roll_funct(ofst,drct,sgm,half,half=True)
		arr_dist.extend(dist)
		arr_sgm.extend(sigmas)
		arr_pos.extend(recon_pos)
	return np.asarray(arr_dist),(np.asarray(arr_sgm)+dim_len),np.asarray(arr_pos)

def cut(dist, err, pos, dist_cut, pos_cut):
	mask_bool = (dist != 0) & (dist < dist_cut) & (np.linalg.norm(pos, axis=1) < pos_cut)  # & (err<2000)
	return dist[mask_bool], err[mask_bool], pos[mask_bool]

def track_dist_group(ofst,drct,sgm,start, end, dim_len=0):
	logger.info('Roll group range: ' + str(start) + ' ' + str(end))
	half = ofst.shape[0]/2
	arr_dist, arr_sgm, arr_pos = [], [], []
	for i in range(start, end+1):
		dist,sigmas,recon_pos = roll_funct(ofst,drct,sgm,i,half=False)
		arr_dist.extend(dist)
		arr_sgm.extend(sigmas)
		arr_pos.extend(recon_pos)
	if ofst.shape[0] & 0x1:
		pass						#condition removed if degeneracy is kept
	else:
		dist,sigmas,recon_pos = roll_funct(ofst,drct,sgm,half,half=True)
		arr_dist.extend(dist)
		arr_sgm.extend(sigmas)
		arr_pos.extend(recon_pos)
	logger.info('Roll group range complete: ' + str(start) + ' ' + str(end))
	logger.info('Total: ' + str(len(arr_dist)))
	dist = np.asarray(arr_dist)
	err = np.asarray(arr_sgm)+dim_len
	rcn_pos = np.asarray(arr_pos)
	return cut(dist, err, rcn_pos, 0.1, 5000)

	#return np.asarray(arr_dist),(np.asarray(arr_sgm)+dim_len),np.asarray(arr_pos)

def track_dist_threaded(ofst,drct,sgm=False,outlier=False,dim_len=0):
	half = ofst.shape[0]/2
	arr_dist, arr_sgm, arr_pos = [], [], []

	pool = Pool(multiprocessing.cpu_count())
	results = []
	count = (ofst.shape[0]-1)/2+1
	photons_per_chunk = 8		# Artificial limitation
	chunks = 10
	chunk_step = count // chunks
	tail_chunk_size = count % chunks
	# Lot's of hacking around in here - need to test..  and clean up
	print('Total rolls: ' + str(count))
	print('Chunks and chunk size: ' + str(chunks) + ' ' + str(chunk_step))
	print('Tail chunk size: ' + str(tail_chunk_size))
	for i in range(1, count, chunk_step):
		range_end = i+photons_per_chunk
		if range_end > count:		# Will this miss the last one?
			result = pool.apply_async(track_dist_group, (ofst, drct, sgm, i, count))
		else:
			result = pool.apply_async(track_dist_group, (ofst, drct, sgm, i, range_end))

		#result = pool.apply_async(roll_funct, (ofst,drct,sgm,i)) # ,half=False,outlier=outlier))
		results.append(result)
		if i % 100 == 0:
			logger.info('Count ' + str(i) + ' of ' + str(count))
		logger.info('1 Result: ' + str(results[0]))
	childs = multiprocessing.active_children()
	logger.info('Child count: ' + str(len(childs)))
	pool.close()
	childs = multiprocessing.active_children()
	logger.info('Child count: ' + str(len(childs)))
	pool.join()
	childs = multiprocessing.active_children()
	logger.info('Child count: ' + str(len(childs)))
	logger.info('Collecting results...')
	for result in results:
		result_value = result.get()
		c_dist = result_value[0]
		c_err = result_value[1]
		c_rcn_pos = result_value[2]
		#c_dist, c_err, c_rcn_pos = cut(dist, err, rcn_pos, 0.1, 5000)
		'''
		mask_bool = (dist != 0) & (dist < 0.01) & (np.linalg.norm(rcn_pos, axis=1) < 5000)  # & (err<2000)
		# mask_bool = np.ones(len(err),dtype=bool)
		c_dist = dist[mask_bool]
		c_err = err[mask_bool]
		c_rcn_pos = rcn_pos[mask_bool]
		'''
		arr_dist.extend(c_dist)
		arr_sgm.extend(c_err)
		arr_pos.extend(c_rcn_pos)

	if ofst.shape[0] & 0x1:
		pass						#condition removed if degeneracy is kept
	else:
		dist,sigmas,recon_pos = roll_funct(ofst,drct,sgm,half,half=True)
		arr_dist.extend(dist)
		arr_sgm.extend(sigmas)
		arr_pos.extend(recon_pos)
	return np.asarray(arr_dist),(np.asarray(arr_sgm)+dim_len),np.asarray(arr_pos)

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('cfg', help='detector configuration')	# Only need one or the other argument
	parser.add_argument('hdf5', help='HDF5 file')
	args = parser.parse_args()
	if args.hdf5 is None:
		cfg = args.cfg
		particle = 'gamma'
		lg = [0,0,0]
		energy = 15.0
		vtx,trx = sim_ev(cfg,particle,lg,energy)
		logger.info('Simulation done, starting reconstruction')
		dist,err,rcn_pos = track_dist(trx.hit_pos.T,trx.means.T,trx.sigmas,trx.lens_rad)
	else:
		with h5py.File(args.hdf5, 'r') as f:
			ks_par = []
			i_idx = 0
			ix = 0  # Just assume one event
			f_idx = f['idx_tr'][ix]
			hit_pos = f['coord'][0, i_idx:f_idx, :]
			means = f['coord'][1, i_idx:f_idx, :]
			sigmas = f['sigma'][i_idx:f_idx]
			dist, err, rcn_pos = track_dist_threaded(hit_pos, means, sigmas, f['r_lens'][()])
			logger.info('Distances computted.  Cutting...')
	'''
	mask_bool = (dist != 0) & (dist < 0.01) & (np.linalg.norm(rcn_pos, axis=1) < 5000)  # & (err<2000)
	# mask_bool = np.ones(len(err),dtype=bool)
	c_rcn_pos = rcn_pos[mask_bool]
	c_err = err[mask_bool]
	c_dist = dist[mask_bool]
	logger.info('Result & cut sizes: ' + str(len(rcn_pos)) + ' ' + str(len(c_rcn_pos)))
	'''
	c_rcn_pos = rcn_pos		# Pre-cut - so just copy them
	c_err = err
	c_dist = dist
	logger.info('Result size: ' + str(len(rcn_pos)))
	logger.info('Saving...')
	with h5py.File('image-'+args.cfg+'.h5','w') as f:
		en_depo = f.create_dataset('pos', data=rcn_pos, chunks=True)
		coord = f.create_dataset('dist', data=dist, chunks=True)
		uncert = f.create_dataset('sigma', data=err, chunks=True)

	logger.info('Plotting...')
	plt.hist(c_err,bins=100)
	plt.show()
	plt.hist(c_dist,bins=100)
	plt.show()
	# print rcn_pos.shape,c_rcn_pos.shape
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.plot(c_rcn_pos[:,0],c_rcn_pos[:,1],c_rcn_pos[:,2],'.', markersize=0.5)
	if args.hdf5 is None:
		ax.plot(vtx[:,0],vtx[:,1],vtx[:,2],'.')
	plt.show()
