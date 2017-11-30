import DetectorResponseGaussAngle as dr
import matplotlib.pyplot as plt
import numpy as np

import paths

def normalize(arr, ax):
	return np.einsum('ij,i->ij',arr,1/np.linalg.norm(arr,axis=ax))

if __name__ == '__main__':
	for s in ['K4_10']:
		arr = []
		cfg = 'cfSam1_%s'%s
		print cfg
		in_file = paths.get_calibration_file_name(cfg)
		det_res = dr.DetectorResponseGaussAngle(cfg,10,10,10,in_file)
		n_lens = det_res.n_lens_sys
		n_pmts_per_surf = det_res.n_pmts_per_surf
		lns_center = det_res.lens_centers
		means = (det_res.means).T
		sigmas = det_res.sigmas
		face_center = np.mean(lns_center.reshape((20,n_lens,3)),axis=1)
		px_lens_means = means.reshape((20,n_lens,n_pmts_per_surf,3))
		px_lens_sigmas = sigmas.reshape((20,n_lens,n_pmts_per_surf))	
		u_dir = np.cross(face_center,np.array([0,0,1]))
		if np.where(np.einsum('ij,ij->i',u_dir,u_dir)<0.00001)[0] == []:
			u_dir = np.cross(face_center[np.where(np.einsum('ij,ij->i',u_dir,u_dir)<0.00001)[0]],np.array([0,1,0]))
		u_dir = normalize(u_dir,1)
		v_dir = normalize(np.cross(face_center, u_dir),1)
		u_proj = np.einsum('ijkl,il->ijk',px_lens_means,u_dir)
		v_proj = np.einsum('ijkl,il->ijk',px_lens_means,v_dir)
		#angle = np.einsum('ijkl,ijkl->ijk',px_lens_means,np.roll(px_lens_means,1,axis=2))[:,:,1:]
		#print u_proj[0,0,angle[0,0,:]<0.9985]
		_,bn,_ = plt.hist(sigmas.flat,bins=100)
		plt.yscale('log', nonposy='clip')
		plt.xlabel('sigma value')
		plt.show()
		cut_val = float(raw_input('cut value: '))
		plt.hist(sigmas.flat,bins=100,color='b')
		plt.hist(sigmas.flat[sigmas.flat>cut_val],bins=bn,color='r')
                plt.yscale('log', nonposy='clip')
                plt.xlabel('sigma value')
                plt.show()

		for i in xrange(n_lens):
			circle = plt.Circle((0,0),1,facecolor='none',edgecolor='r')
			fig = plt.gcf()
			ax = fig.gca()
			ax.add_artist(circle)
			ax.scatter(u_proj[0,i],v_proj[0,i],c='b')
			ax.scatter(u_proj[0,i,px_lens_sigmas[0,i,:]>cut_val],v_proj[0,i,px_lens_sigmas[0,i,:]>cut_val],c='r',s=50)
			ax.set_xlim(-1,1)
			ax.set_ylim(-1,1)
			ax.set_aspect('equal')
			plt.show()
		'''if s[0] == 'M':
			bn = 1000
		elif s[0] == 'K':
			bn = 100
		elif s[0] == 'k':
			bn = 50
		plt.hist(arr,bins=bn)
		plt.yscale('log', nonposy='clip')
		plt.show()'''
