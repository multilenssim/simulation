import iter_analysis as ia
import h5py,glob,argparse
import numpy as np


def clusterize(fname, ev):
	lb = []
	from sklearn.cluster import DBSCAN
	with h5py.File(fname,'r') as f:
		i_idx = 0
		for ix in xrange(ev):
			f_idx = f['idx_depo'][ix]
			vert = f['en_depo'][i_idx:f_idx,:]
			i_idx = f_idx
			db = DBSCAN(eps=3, min_samples=10).fit(vert)
			labels =  db.labels_
			labels = labels[labels!=-1]
			lb.append(max(labels))
	return lb

def plot_cluster():
	from mpl_toolkits.mplot3d import Axes3D
	import matplotlib.pyplot as plt
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.plot(vert[:,0],vert[:,1],vert[:,2],'.',color='red')
	plt.show()
	plt.close()
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(set(labels)))]
	for lb,col in zip(set(labels),colors):
		ax.plot(vert[labels==lb][:,0],vert[labels==lb][:,1],vert[labels==lb][:,2],'.',markerfacecolor=tuple(col))
	plt.show()

def track_hist(fname,ev):
        with h5py.File(fname,'r') as f:
		ks_par = []
                i_idx = 0
                for ix in xrange(ev):
                        f_idx = f['idx_tr'][ix]
                        hit_pos = f['coord'][0,i_idx:f_idx,:]
			means = f['coord'][1,i_idx:f_idx,:]
			sigmas = f['sigma'][i_idx:f_idx]
                        i_idx = f_idx
			tr_dist, er_dist = ia.track_dist(hit_pos, means, sigmas, False, f['r_lens'][()])
			err_dist = 1./np.asarray(er_dist)
			ks_par.append(ia.make_hist(bn_arr, tr_dist, err_dist))
		return ks_par


if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('path', help='insert path-to-file with seed location')
	args = parser.parse_args()
	path = args.path
	n_ev = 500
	max_val = 2000
	bin_width = 10
	n_bin = max_val/bin_width
	bn_arr = np.linspace(0,max_val,n_bin)
	for fname in sorted(glob.glob(path+'*sim.h5')):
		print fname
		chi2_arr = track_hist(fname,n_ev)
	        if fname[len(path)] == 'e':
			n_null = [np.mean(chi2_arr,axis=0),np.std(chi2_arr,axis=0)]
			e_hist = ia.chi2(n_null,chi2_arr)
		elif fname[len(path)] == 'g':
                        g_hist = ia.chi2(n_null,chi2_arr)
	np.savetxt(path+'electron-gammac2',(e_hist,g_hist))
