import DetectorResponseGaussAngle as dr
import matplotlib.pyplot as plt
import numpy as np
import paths

def normalize(arr, ax):
	return np.einsum('ij,i->ij',arr,1/np.linalg.norm(arr,axis=ax))

def proj(cfg):
	in_file = paths.get_calibration_file_name(cfg)
	det_res = dr.DetectorResponseGaussAngle(cfg,10,10,10,in_file)
	n_lens = det_res.n_lens_sys
	n_pmts_per_surf = det_res.n_pmts_per_surf
	lns_center = det_res.lens_centers
	means = (det_res.means).T
	sigmas = det_res.sigmas
	px_lens_means = means.reshape((n_lens,n_pmts_per_surf,3))
	px_lens_sigmas = sigmas.reshape((n_lens,n_pmts_per_surf))
	u_dir = np.cross(lns_center,[0,0,1])
	mask = np.all(u_dir==0,axis=1)
	u_dir[mask] = np.cross(lns_center[mask],[0,1,0])
	u_dir = normalize(u_dir,1)
	v_dir = normalize(np.cross(lns_center, u_dir),1)
	u_proj = np.einsum('ijk,ik->ij',px_lens_means,u_dir)
	v_proj = np.einsum('ijk,ik->ij',px_lens_means,v_dir)
	return px_lens_means, px_lens_sigmas, u_proj, v_proj


if __name__ == '__main__':
	cut = True 
	for s in ['K200_10']:
		cfg = 'cfSam1_%s'%s
		print cfg
		#angle = np.einsum('ijkl,ijkl->ijk',px_lens_means,np.roll(px_lens_means,1,axis=2))[:,:,1:]
		#print u_proj[0,0,angle[0,0,:]<0.9985]
		px_lens_means, px_lens_sigmas, u_proj, v_proj = proj(cfg)
		sin_dir = np.linalg.norm(np.asarray([u_proj.flatten(),v_proj.flatten()]),axis=0)
		_,bn,_ = plt.hist(np.arcsin(sin_dir),bins=100)
		asn = np.arcsin(sin_dir).reshape((u_proj.shape))
		plt.yscale('log', nonposy='clip')
		plt.xlabel('calibrated pixel angular aperture')
		plt.show()
		if cut:
			cut_val = np.fromstring(raw_input('cut value: '),dtype=float,sep=' ')
			#plt.hist(px_lens_sigmas.flat,bins=100,color='b')
			#plt.hist(px_lens_sigmas.flat[px_lens_sigmas.flat>cut_val],bins=bn,color='r')
                	#plt.yscale('log', nonposy='clip')
                	#plt.xlabel('sigma value')
                	#plt.show()

		for i in np.random.choice(u_proj.shape[0],2,replace=False):
			circle = plt.Circle((0,0),1,facecolor='none',edgecolor='r')
			fig = plt.gcf()
			ax = fig.gca()
			ax.add_artist(circle)
			ax.scatter(u_proj[i],v_proj[i],c='b')
			if cut:
				ax.scatter(u_proj[i,asn[i]<cut_val[1]],v_proj[i,asn[i]<cut_val[1]],c='r',s=50)
				ax.scatter(u_proj[i,asn[i]<cut_val[0]],v_proj[i,asn[i]<cut_val[0]],c='g',s=50)
			ax.set_xlim(-1,1)
			ax.set_ylim(-1,1)
			ax.set_aspect('equal')
			plt.show()
