import DetectorResponseGaussAngle as dr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import paths

def normalize(arr, ax):
	return np.einsum('ij,i->ij',arr,1/np.linalg.norm(arr,axis=ax))

def proj(cfg,cl):
	in_file = paths.get_calibration_file_name(cfg,cl)
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
	for s in ['K200_8_small']:
		cfg = 'cfSam1_%s'%s
		cal = '_narrow'
		print cfg
		px_lens_means, px_lens_sigmas, u_proj, v_proj = proj(cfg,cal)
		sin_dir = np.linalg.norm([u_proj.flatten(),v_proj.flatten()],axis=0)
		_,bn,_ = plt.hist(np.arcsin(sin_dir),bins=100)
		#asn = np.arcsin(sin_dir).reshape((u_proj.shape))
		plt.yscale('log', nonposy='clip')
		plt.xlabel('calibrated pixel angular aperture')
		plt.show()
		if cut:
			plt.hist(px_lens_sigmas.flat,bins=100,color='b')
                	plt.yscale('log', nonposy='clip')
                	plt.xlabel('sigma value')
                	plt.show()
                        cut_val = np.fromstring(raw_input('cut value: '),dtype=float,sep=' ')
			colors = cm.rainbow(np.linspace(0,1,len(cut_val)))

		for i in np.random.choice(u_proj.shape[0],4,replace=False):
			circle = plt.Circle((0,0),1,facecolor='none',edgecolor='r')
			fig = plt.gcf()
			ax = fig.gca()
			ax.add_artist(circle)
			ax.scatter(u_proj[i],v_proj[i],c='b')
			if cut:
				for cv,cl in reversed(zip(cut_val,colors)):
					ax.scatter(u_proj[i,px_lens_sigmas[i]<cv],v_proj[i,px_lens_sigmas[i]<cv],c=cl,s=50)
			ax.set_xlim(-0.5,0.5)
			ax.set_ylim(-0.5,0.5)
			ax.set_aspect('equal')
			plt.show()
