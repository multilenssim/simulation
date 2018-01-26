from chroma.generator import vertex
import pickle,time,argparse
import nog4_sim as setup
import numpy as np
import paths

'''                        db = DBSCAN(eps=3, min_samples=10).fit(vert)
                        label =  db.labels_
                        labels = label[label!=-1]
                        vert = vert[label!=-1]
                        unique, counts = np.unique(labels, return_counts=True)
                        main_cluster = vert[labels==unique[np.argmax(counts)],:]
'''


def sim_ev(particle,lg,energy):
	gun = vertex.particle_gun([particle], vertex.constant(lg), vertex.isotropic(), vertex.flat(energy*0.999, energy*1.001))
	for ev in sim.simulate(gun,keep_photons_beg=True, keep_photons_end=True, run_daq=False, max_steps=100):
		vert = ev.photons_beg.track_tree
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
	sm = np.stack((sgm,r_sgm),axis=1)
	off_stack = np.stack((ofst,r_ofst),axis=1)
	drct_stack = np.stack((drct,r_drct),axis=1)
	multp = syst_solve(drct,r_drct,ofst_diff)
	#multp[multp==0] = 1
	sigmas = np.linalg.norm(np.einsum('ij,ij->ij',sm,multp),axis=1)
	poc = off_stack + np.einsum('ijk,ij->ijk',drct_stack,multp)
	dist = poc[:,0,:] - poc[:,1,:]
	return np.linalg.norm(dist,axis=1),sigmas,np.mean(poc,axis=1)

def track_dist(ofst,drct,sgm,dim_len=0):
	half = ofst.shape[0]/2
	arr_dist, arr_sgm, arr_pos = [], [], []
	for i in range(1,(ofst.shape[0]-1)/2+1):
		dist,sigmas,recon_pos = roll_funct(ofst,drct,sgm,i,half=False)
		mask_bool = (dist!=0) & (dist<0.15) & (np.linalg.norm(recon_pos,axis=1)<700)
		arr_dist.extend(dist[mask_bool])
		arr_sgm.extend(sigmas[mask_bool])
		arr_pos.extend(recon_pos[mask_bool])
	if ofst.shape[0] & 0x1: pass						#condition removed if degeneracy is kept
	else:
		dist,sigmas,recon_pos = roll_funct(ofst,drct,sgm,half,half=True)
                mask_bool = (dist!=0) & (dist<0.15) & (np.linalg.norm(recon_pos,axis=1)<700)
                arr_dist.extend(dist[mask_bool])
                arr_sgm.extend(sigmas[mask_bool])
	return np.asarray(arr_dist),(np.asarray(arr_sgm)+dim_len),np.asarray(arr_pos)


cfg = 'cfSam1_K200_8_small'
sim,analyzer = setup.sim_setup(cfg,paths.get_calibration_file_name(cfg),useGeant4=True)
print 'Configuration loaded'
lg = [0,0,0]
energy = 2.0
dct = {}
for i in xrange(1):
	n_ph,s_coord = [],[]
	particle = np.random.choice(['neutron','neutron'])
	vtx,trx = sim_ev(particle,lg,energy)
	for vx in vtx:
		try:
			n_ph.append(vtx[vx]['child_processes']['Scintillation'])
			s_coord.append(vtx[vx]['position'])
		except KeyError:
			pass
	dct['scint%i'%i] = {'n_photons':n_ph,'coord':s_coord}
	print 'Simulation done, starting reconstruction'
	dist,err,rcn_pos = track_dist(trx.hit_pos.T,trx.means.T,trx.sigmas,trx.lens_rad)
	dct['vtx%i'%i] = rcn_pos

with open('dataset_%s.p'%particle,'w') as f:
	pickle.dump(dct,f,pickle.HIGHEST_PROTOCOL)
