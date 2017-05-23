from scipy.stats import chisquare
import first_test_jacopo as jacopo
import matplotlib.pyplot as plt
import kabamland2 as kbl
import numpy as np
import time

def fixed_dist_hist(dist,sample,bn_arr,amount,sim,analyzer,sgm=False,plot=False,sigma=0.01,reth=False):
	ks_par = []
	locs1, locs2, rad = jacopo.fixed_dist(sample,5000,rads=dist)
	if plot:
		jacopo.plot_sphere(locs1,loc2=locs2,rads=rad)
	for lc1,lc2 in zip(locs1,locs2):
		sim_events = jacopo.create_double_source_events(lc1, lc2, sigma, amount/2, amount/2)
		for ev in sim.simulate(sim_events, keep_photons_beg = True, keep_photons_end = True, run_daq=False, max_steps=100):
			tracks = analyzer.generate_tracks(ev)
		if sgm:
			tr_dist,er_dist = jacopo.track_dist(tracks.hit_pos.T,tracks.means.T,sgm=tracks.sigmas)
			err_dist = 1./np.asarray(er_dist)
		else:
			tr_dist = jacopo.track_dist(tracks.hit_pos.T,tracks.means.T)
			err_dist = np.ones(len(tr_dist))
		#c_hist = np.cumsum(bkg_hist - jacopo.make_hist(bn_arr,tr_dist,err_dist))
		#ks_par.append(c_hist[np.argmax(np.abs(c_hist))])
		#ks_par.append(np.dot(jacopo.make_hist(bn_arr,tr_dist,err_dist),bn_arr))
		if reth: ks_par.append(jacopo.make_hist(bn_arr,tr_dist,err_dist))
		else: ks_par.append(np.average(tr_dist,weights=err_dist))
	return ks_par

def bkg_dist_hist(sample,bn_arr,amount,sim,analyzer,sgm=False,plot=False,sigma=0.01):
	ks_par,loc_p,loc_m = [],[],[]
	i = 0
	location = jacopo.sph_scatter(sample,in_shell=0,out_shell=5000)
	if plot:
		jacopo.plot_sphere(location)
	for lg in location:
		i += 1
		sim_event = kbl.gaussian_sphere(lg, sigma, amount)
		for ev in sim.simulate(sim_event, keep_photons_beg = True, keep_photons_end = True, run_daq=False, max_steps=100):
			tracks = analyzer.generate_tracks(ev)
		if sgm:
			tr_dist,er_dist = jacopo.track_dist(tracks.hit_pos.T,tracks.means.T,sgm=tracks.sigmas)
			err_dist = 1./np.asarray(er_dist)
		else:
			tr_dist = jacopo.track_dist(tracks.hit_pos.T,tracks.means.T)
			err_dist = np.ones(len(tr_dist))
		#c_hist = np.cumsum(bkg_hist - jacopo.make_hist(bn_arr,tr_dist,err_dist))
		#ks_par.append(c_hist[np.argmax(np.abs(c_hist))])
		#ks_par.append(np.dot(jacopo.make_hist(bn_arr,tr_dist,err_dist),bn_arr))
		ks_par.append(np.average(tr_dist,weights=err_dist))
		if i%200 == 0:
			print i
	return ks_par
	
def chi2(bkg_hist,chi2h):
	c2hist = []
	for c2 in chi2h:
		c2hist.append(chisquare(c2,f_exp=bkg_hist)[0])
	return c2hist

def plot_cl(ks_bin,c_bkg,c_sgn):
	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()
	ax1.plot(ks_bin, c_bkg, 'r-')
	ax2.plot(ks_bin, 1-c_sgn, 'b-')
	ax1.set_ylim([0, 1.05])
	ax2.set_ylim([0, 1.05])
	ax1.set_xlabel('cut value')
	ax1.set_ylabel('signal efficiency', color='r')
	ax2.set_ylabel('background discrimination', color='b')
	plt.axhline(0.95, xmin=0.05,xmax=0.95, color='b', linestyle='dashed', linewidth=2)
	plt.text(ks_bin[3],0.92,'95%')
	plt.title('%i mm distance for the double site' %dist)
	plt.show()

def use_avg():
	ks_par_bkg = bkg_dist_hist(sample,bn_arr,6600,sim,analyzer,sgm=sgm)
	ks_par = fixed_dist_hist(dist,sample,bn_arr,6600,sim,analyzer,sgm=sgm)
	ks_bin = np.linspace(min(ks_par_bkg),max(ks_par),50)
	ks_hist_bkg = jacopo.make_hist(ks_bin,ks_par_bkg,np.ones(sample))
	ks_hist = jacopo.make_hist(ks_bin,ks_par,np.ones(sample))
	plot_cl(ks_bin,np.cumsum(ks_hist_bkg),np.cumsum(ks_hist))

def use_chi2():
	bkg_hist,chi2h = jacopo.band_shell_bkg(sample,bn_arr,6600,sim,analyzer,0,5000,sgm=sgm)
	ks_par = fixed_dist_hist(dist,sample,bn_arr,6600,sim,analyzer,sgm=sgm,reth=True)
	ks_bin = np.linspace(0,np.max(ks_par),200)
	sng_h = jacopo.make_hist(ks_bin,chi2(bkg_hist,chi2h),np.ones(sample))
	bkg_h = jacopo.make_hist(ks_bin,chi2(bkg_hist,ks_par),np.ones(sample))
	plt.plot(ks_bin,sng_h)
	plt.plot(ks_bin,bkg_h)
	plt.show()
	plt.close()
	plot_cl(ks_bin,np.cumsum(sng_h),np.cumsum(bkg_h))

max_val = 2000
bin_width = 10
n_bin = max_val/bin_width
sample = 100
dist = 200
sgm = True
bn_arr = np.linspace(0,max_val,n_bin)
sim,analyzer = jacopo.sim_setup('cfJiani3_4','/home/miladmalek/TestData/detresang-cfJiani3_4_1DVariance_noreflect_100million.root')
#use_avg()
use_chi2()



#('cfSam1_1','/home/miladmalek/TestData/detresang-cfSam1_1_1DVariance_noreflect_100million.root')
'''
def find_cl(arr,c_arr,val):
	idx = np.abs(c_arr - val).argmin()
	return arr[idx]

def plot_histo():
	plt.plot(ks_bin,ks_hist_bkg,label='null hypothesis',ls='steps')
	plt.plot(ks_bin,ks_hist,label='%i mm distance'%dist,ls='steps')
	plt.axvline(find_cl(ks_bin,c_bkg,0.68), color='b', linestyle='dashed', linewidth=2)
	plt.text(find_cl(ks_bin,c_bkg,0.68)+0.0001,0.05,'68%CL',rotation=90)
	plt.axvline(find_cl(ks_bin,c_bkg,0.9), color='b', linestyle='dashed', linewidth=2)
	plt.text(find_cl(ks_bin,c_bkg,0.9)+0.0001,0.05,'90%CL',rotation=90)
	plt.axvline(find_cl(ks_bin,c_bkg,0.95), color='b', linestyle='dashed', linewidth=2)
	plt.text(find_cl(ks_bin,c_bkg,0.95)+0.0001,0.05,'95%CL',rotation=90)
	plt.legend()
	plt.show()
	plt.close()
'''
