import nog4_sim as setup
from mpl_toolkits.mplot3d import Axes3D
from chroma.generator import vertex
from chroma.event import Photons
import matplotlib.pyplot as plt
import config_stat
import numpy as np
import kabamland2
import pickle
import paths

def sim_ev(cfg,particle,lg,energy):
	print 'Configuration loaded'
	if particle == None:
		sim,analyzer = setup.sim_setup(cfg,paths.get_calibration_file_name(cfg,cl),useGeant4=False)
		det_res = analyzer.det_res
		energy = int(energy*8000)
		gun = setup.create_double_source_events(np.asarray([0,0,0]), np.asarray([0,0,0]), 0.01, energy/2,energy/2)
		#gun = kabamland2.gaussian_sphere(lg, 0.01, energy)
		#gun = gen_test(20,1,det_res)
	else:
		sim,analyzer = setup.sim_setup(cfg,paths.get_calibration_file_name(cfg,cl),useGeant4=True)
		det_res = analyzer.det_res
		gun = vertex.particle_gun([particle], vertex.constant(lg), vertex.isotropic(), vertex.flat(energy*0.999, energy*1.001))
	for ev in sim.simulate(gun,keep_photons_beg=True, keep_photons_end=True, run_daq=False, max_steps=100):
		vert = ev.photons_beg.pos
		tracks,arr_bin = analyzer.generate_tracks(ev,qe=(1./3.),heat_map=True)
		pixel_loc = det_res.pmt_bin_to_position(arr_bin)
	return vert, tracks, pixel_loc, det_res, arr_bin

def gen_test(n,r,det_res):
	pos = np.random.normal(0,0.01,(n*det_res.n_lens_sys,3))
	if r == 0.0:
		lns_dir = config_stat.normalize(det_res.lens_centers,1)
		drct = np.repeat(lns_dir,n,axis=0)+np.random.normal(0.0,0.001,(n*det_res.n_lens_sys,3))
	else:
		u_proj = config_stat.normalize(np.cross(det_res.lens_centers,[0,0,1]),1)
		v_proj = config_stat.normalize(np.cross(det_res.lens_centers,u_proj),1)
		angle = 2*np.pi*np.random.rand(n*det_res.n_lens_sys)
		drct = np.repeat(det_res.lens_centers,n,axis=0) + r*np.einsum('i,ij->ij',np.sin(angle),np.repeat(u_proj,n,axis=0)) + r*np.einsum('i,ij->ij',np.cos(angle),np.repeat(v_proj,n,axis=0))
	pol = np.cross(drct,np.random.rand(n*det_res.n_lens_sys,3))
	wavelengths = np.repeat(300.0,n*det_res.n_lens_sys)
	return Photons(pos,drct,pol,wavelengths)

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
	sigmas = np.linalg.norm(np.einsum('ij,ij->ij',sm,multp),axis=1)
	poc = off_stack + np.einsum('ijk,ij->ijk',drct_stack,multp)
	dist = poc[:,0,:] - poc[:,1,:]
	return np.linalg.norm(dist,axis=1),sigmas,np.mean(poc,axis=1)

def track_dist(ofst,drct,sgm,dim_len=0):
	half = ofst.shape[0]/2
	arr_dist, arr_sgm, arr_pos, arr_px = [], [], [], []
	for i in range(1,(ofst.shape[0]-1)/2+1):
		dist,sigmas,recon_pos = roll_funct(ofst,drct,sgm,i,half=False)
		arr_dist.extend(dist)
		arr_sgm.extend(sigmas)
		arr_pos.extend(recon_pos)
	if not ofst.shape[0] & 0x1:						#condition removed if degeneracy is kept
		dist,sigmas,recon_pos = roll_funct(ofst,drct,sgm,half,half=True)
		arr_dist.extend(dist)
		arr_sgm.extend(sigmas)
		arr_pos.extend(recon_pos)
	return np.asarray(arr_dist),(np.asarray(arr_sgm)+dim_len),np.asarray(arr_pos)

def proj(l_center,dct,arr_bin,det_res):
	l_number = arr_bin/det_res.n_pmts_per_surf
	lns_list = np.unique(l_number,axis=0)
	#lens_label = arr_bin/det_res.n_pmts_per_surf
	r_means = (det_res.means.T).reshape((-1,det_res.n_pmts_per_surf,3))
	for ll in lns_list[:2]:
		mask = l_number==ll
		lens_dir = l_center[mask][0]
		l_dct = dct[mask].reshape((-1,3))
		u_dir = np.cross(lens_dir,[0,0,1])
		if np.array_equal(u_dir,[0,0,0]):
			u_dir = np.cross(lens_dir,[0,1,0])
		u_dir = u_dir/np.linalg.norm(u_dir)
		v_dir = np.cross(lens_dir,u_dir)/np.linalg.norm(np.cross(lens_dir,u_dir))
		u_proj = np.einsum('ij,j->i',l_dct,u_dir)
		v_proj = np.einsum('ij,j->i',l_dct,v_dir)
		u_tot = np.einsum('ij,j->i',r_means[ll],u_dir)
		v_tot = np.einsum('ij,j->i',r_means[ll],v_dir)
		print u_tot.shape
                plt.xlim(-0.5,0.5)
                plt.ylim(-0.5,0.5)
		plt.scatter(u_tot,v_tot,s=20)
		plt.scatter(u_proj,v_proj,s=40)
		plt.show()
		plt.clf()



if __name__ == '__main__':
	cfg = 'cfSam1_K200_8_small'
	cl = ''
	particle = None
	lg = np.asarray([500,0,0])
	energy = 2.0
	vtx,trx,p_loc,det_res,arr_bin = sim_ev(cfg,particle,lg,energy)
	print 'Simulation done, starting reconstruction'
	proj(trx.hit_pos.T,trx.means.T,arr_bin,det_res)
	plt.close()
	exit()
	dist,err,rcn_pos = track_dist(trx.hit_pos.T,trx.means.T,trx.sigmas,trx.lens_rad)
	mask_bool = (np.linalg.norm(rcn_pos,axis=1)<600) & (dist!=0) & (dist<0.15)
	#mask_bool = np.ones(len(dist), dtype=bool)
	c_rcn_pos = rcn_pos[mask_bool]
	#c_err = err[mask_bool]
	#c_dist = dist[mask_bool]
	plt.hist(np.linalg.norm(c_rcn_pos,axis=1),bins=1000)
	plt.xlabel('radial position of the reconstructed mid-point (not r$^2$ normalized) [mm]')
	plt.show()
	#exit()
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	#for p,d in zip(trx.hit_pos.T,trx.means.T):
	#	ln = p+300*d
	#	plt.plot([p[0],ln[0]],[p[1],ln[1]],[p[2],ln[2]],color='b',linewidth=0.5)
	#ax.plot(focal_center[:,0],focal_center[:,1],focal_center[:,2],'.',markersize=5)
	ax.plot(c_rcn_pos[:,0],c_rcn_pos[:,1],c_rcn_pos[:,2],'.', markersize=0.5)
	ax.plot(vtx[:,0],vtx[:,1],vtx[:,2],'.')
	ax.set_xlim(-8000, 8000)
	ax.set_ylim(-8000, 8000)
	ax.set_zlim(-8000, 8000)
	plt.show()
