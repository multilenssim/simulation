import time,os,argparse
import numpy as np
import lensmaterials as lm
import kabamland2 as kbl
import matplotlib.pyplot as plt
import pylab
from EventAnalyzer import EventAnalyzer
from chroma.detector import Detector
from chroma.sim import Simulation
from chroma.sample import uniform_sphere
from chroma.loader import load_bvh
from DetectorResponseGaussAngle import DetectorResponseGaussAngle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle


# To run this script: python first_test_jacopo.py 1 will plot the simulation of one track distance distribution from one source
# python first_test_jacopo.py 2 will run the simulation with two sources with various separation and relative amount
# add the flag -save to save the histogram without plotting it

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


def plot_sphere(loc1,loc2=None,rads=None,cl=None):
	u = np.linspace(0, 2 * np.pi, 100)
	v = np.linspace(0, np.pi, 100)
	x = np.outer(np.cos(u), np.sin(v))
	y = np.outer(np.sin(u), np.sin(v))
	z = np.outer(np.ones(np.size(u)), np.cos(v))
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	if loc2 == None:
		ax.plot(loc1[:,0]/5000,loc1[:,1]/5000,loc1[:,2]/5000,'.')
		if cl != None:
			ax.plot(cl[:,0]/5000,cl[:,1]/5000,cl[:,2]/5000,'.')
	else:
		color = plt.cm.rainbow(np.linspace(0, 1, len(rads)))
		for i,j in zip(loc1/5000,loc2/5000):
			ax.plot([i[0],j[0]],[i[1],j[1]],[i[2],j[2]],'.')
	ax.plot_surface(x,y,z,linewidth=0.2,alpha=0.1)
	plt.show()
	plt.close()

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

def track_dist(ofst,drct,sgm=None):
	half = ofst.shape[0]/2
	arr_dist, arr_sgm = [], []
	for i in range(1,(ofst.shape[0]-1)/2+1):
		ofst_diff = ofst - np.roll(ofst,i,axis=0)
		r_drct = np.roll(drct,i,axis=0)
		b_drct = np.cross(drct,r_drct)
		norm_d = b_drct/np.linalg.norm(b_drct,axis=1).reshape(-1,1)
		dist = np.absolute(np.einsum('ij,ij->i',ofst_diff,norm_d))
		remove_nan(dist,ofst_diff,drct)
		arr_dist.extend(dist)
		if sgm != None:
			r_sgm = np.roll(sgm,i,axis=0)
			sm = np.stack((sgm,r_sgm),axis=1)
			multp = syst_solve(drct,r_drct,ofst_diff)
			multp[np.where(multp==0)] = 1
			arr_sgm.extend(np.linalg.norm(np.einsum('ij,ij->ij',sm,multp),axis=1))
	if ofst.shape[0] & 0x1: pass						#condition removed if degeneracy is kept
	else:
		ofst_diff = ofst[:half]-np.roll(ofst,half,axis=0)[:half]
		r_drct = np.roll(drct,half,axis=0)[:half]
		b_drct = np.cross(drct[:half],r_drct)
		norm_d = b_drct/np.linalg.norm(b_drct,axis=1).reshape(-1,1)
		dist = np.absolute(np.einsum('ij,ij->i',ofst_diff,norm_d))
		remove_nan(dist,ofst_diff,drct[:half])
		arr_dist.extend(dist)
		if sgm != None:
			r_sgm = np.roll(sgm,half,axis=0)[:half]
			sm = np.stack((sgm[:half],r_sgm),axis=1)
			multp = syst_solve(drct[:half],r_drct,ofst_diff)
			multp[np.where(multp==0)] = 1
			arr_sgm.extend(np.linalg.norm(np.einsum('ij,ij->ij',sm,multp),axis=1))
	if sgm != None:
		arr_sgm = np.asarray(arr_sgm)
		return arr_dist,arr_sgm
	else:
		return arr_dist

def plot_hist(arr_dist,out_print,save=None,directory=None):
	tot_tracks = len(arr_dist)
	plt.hist(np.asarray(arr_dist),bins=1000,normed=True)
	plt.xlim((0,7500))
	plt.ylim((1e-7,1e-2))
	plt.xlabel('reciprocal distance [mm]')
	plt.title('Distance distribution for %s with a total of %i entries'%(out_print,tot_tracks))
	plt.yscale('log')
	plt.subplots_adjust(left=0.03, right=0.99, top=0.97, bottom=0.06)
	if save == False:
		plt.show()
	else:
		F = pylab.gcf()
		ds = F.get_size_inches()
		F.set_size_inches((ds[0]*2.45,ds[1]*1.83))
		if directory == None:
			F.savefig(out_print.replace(' ','')+'.png')
		else:
			F.savefig(directory+out_print.replace(' ','')+'.png')

def make_hist(bn_arr,arr,c_wgt,norm=True):
	wgt = []
	np_double = np.asarray(arr)
	bn_wdt = bn_arr[1] - bn_arr[0]
	for bn in bn_arr:
		wgt.extend(np.dot([(np_double>=bn) & (np_double<(bn+bn_wdt))],c_wgt))
	if norm:
		return np.asarray(wgt)/abs(sum(wgt))
	else:
		return np.asarray(wgt)


def substract_hist(bn_arr,bkg,sgnl,sigma_sgnl,dists):
	dist = len(dists)
	color = plt.cm.rainbow(np.linspace(0, 1, dist))
	rct, label = [],[]
	for i in reversed(xrange(dist)):
		plt.fill_between(bn_arr,sgnl[i]-bkg[0]+sigma_sgnl[i],sgnl[i]-bkg[0]-sigma_sgnl[i],color=color[i])
		rct.append(Rectangle((0, 0), 1, 1, fc=color[i]))
		label.append('respective distance %0.2f mm'%dists[i])
	plt.fill_between(bn_arr,bkg[1],-bkg[1])
	plt.xlabel('respective distance between tracks [mm]')
	plt.ylabel('normalized residuals (compared to SS)')
	plt.title('Spherical shell 0-1000mm')
	plt.legend(rct,label)
	plt.xlim(0,3000)
	plt.ylim(-0.04,0.02)
	plt.subplots_adjust(left=0.06, right=0.99, top=0.97, bottom=0.06)	
	plt.show()
	plt.close()

def avg_hist(ss_arr_dist,bn_arr,sgm=None):
	wgt = []
	for arr,i in zip(ss_arr_dist,range(len(ss_arr_dist))):
		if sgm != None:
			c_wgt = sgm[i]
		else:
			c_wgt = np.ones(len(arr))
		wgt.append(make_hist(bn_arr,arr,c_wgt))
	wgt = np.asarray(wgt)
	av_wgt = np.mean(wgt,axis=0)
	s_wgt = np.std(wgt,axis=0)
	return av_wgt,s_wgt

def overlap_hist(arr_dist,double_arr_dist,bn_arr,radius):
	plt.hist(np.asarray(arr_dist),bins=bn_arr,normed=True,label='one source',histtype='step')
	for lab,arr in zip(radius,double_arr_dist):
		plt.hist(np.asarray(arr),bins=bn_arr,normed=True,label='respective distance %0.0f mm'%lab,histtype='step')
	plt.xlabel('respective distance between tracks [mm]')
	plt.legend()
	plt.subplots_adjust(left=0.03, right=0.99, top=0.97, bottom=0.06)	
	plt.show()
	plt.close()

def sim_setup(config,in_file):
	kabamland = Detector(lm.ls)
	kbl.build_kabamland(kabamland, config)
	kabamland.flatten()
	kabamland.bvh = load_bvh(kabamland)
	sim = Simulation(kabamland)
	det_res = DetectorResponseGaussAngle(config,10,10,10,in_file)
	analyzer = EventAnalyzer(det_res)
	return sim, analyzer

def band_shell_bkg(sample,bn_arr,amount,sim,analyzer,i_shell,o_shell,sgm=False,plot=False,sigma=0.01):
	arr_dist = np.zeros(len(bn_arr))
	arr_sgm, chi2 = [], []
	location = sph_scatter(sample,in_shell=i_shell,out_shell=o_shell)
	if plot:
		plot_sphere(location)
	i = 0
	for lg in location:
		sim_event = kbl.gaussian_sphere(lg, sigma, amount)
		for ev in sim.simulate(sim_event, keep_photons_beg = True, keep_photons_end = True, run_daq=False, max_steps=100):
			tracks = analyzer.generate_tracks(ev)
		if sgm:
			tr_dist,err_dist = track_dist(tracks.hit_pos.T,tracks.means.T,sgm=tracks.sigmas)
			arr_sgm = 1./np.asarray(err_dist)
		else:
			tr_dist = track_dist(tracks.hit_pos.T,tracks.means.T)
			arr_sgm = np.ones(len(tr_dist))
		one_ev_hist = make_hist(bn_arr,tr_dist,arr_sgm)
		chi2.append(one_ev_hist)
		arr_dist = np.stack((one_ev_hist,arr_dist))
		arr_dist = np.sum(arr_dist,axis=0)
	print 'ss events produced'
	return arr_dist/sample, chi2

def band_shell_sgn(r_dist,sample,bn_arr,amount,sim,analyzer,sgm=False,plot=False,sigma=0.01):
	av_hist, sigma_hist = [], []
	dists = np.linspace(50,500,r_dist)
	for rads in dists:
		locs1, locs2, rad = fixed_dist(sample,5000,rads=rads)
		av_dist_hist, av_sgm_hist = [], []
		if plot:
			plot_sphere(locs1,loc2=locs2,rads=rad)
		for lc1,lc2 in zip(locs1,locs2):
			sim_events = create_double_source_events(lc1, lc2, sigma, amount/2, amount/2)
			for ev in sim.simulate(sim_events, keep_photons_beg = True, keep_photons_end = True, run_daq=False, max_steps=100):
				tracks = analyzer.generate_tracks(ev)
			if sgm:
				tr_dist,err_dist = track_dist(tracks.hit_pos.T,tracks.means.T,sgm=tracks.sigmas)
				av_sgm_hist.append(1./np.asarray(err_dist))
			else:
				tr_dist = track_dist(tracks.hit_pos.T,tracks.means.T)
				av_sgm_hist.append(np.ones(len(tr_dist)))
			av_dist_hist.append(tr_dist)
		h, h_sigma = avg_hist(av_dist_hist,bn_arr,av_sgm_hist)
		av_hist.append(h)
 		sigma_hist.append(h_sigma)
		print 'dist %dmm done'%rads
	return np.asarray(av_hist), np.asarray(sigma_hist), dists
